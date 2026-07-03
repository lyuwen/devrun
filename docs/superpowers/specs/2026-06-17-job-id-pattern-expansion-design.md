# Job ID Pattern Expansion Design

**Date:** 2026-06-17  
**Status:** Approved  
**Author:** AI Assistant + User

## Problem Statement

Currently, when configuring SWE-bench agentic tasks with the `job_ids` parameter, users must provide a comma-separated list of individual job IDs. For workflows with many sequential or semi-sequential IDs, this becomes tedious and error-prone.

Example:
```yaml
params:
  job_ids: "172.16.1.157,172.16.1.158,172.16.1.159,172.16.1.160,172.16.1.161,172.16.1.162,172.16.1.163"
```

**Goal:** Support pattern expansion so users can write:
```yaml
params:
  job_ids: "172.16.1.[157-163]"
```

## Scope

**In scope:**
- Pattern expansion for numeric ranges with auto-detected padding
- Explicit list syntax for non-sequential IDs
- Mixed ranges and lists within a single bracket
- Reusable utility for potential future use in other tasks

**Out of scope:**
- Wildcard matching against database or filesystem
- Bash-style `{...}` brace expansion (we use `[...]` syntax)
- Non-numeric ranges (e.g., `[a-z]`)
- Nested brackets or complex expressions

## Pattern Syntax

### 1. Range Syntax

**Format:** `prefix[start-end]suffix`

Expands to all integers from `start` to `end` (inclusive). Padding is auto-detected from the first number.

**Examples:**
- `172.16.1.[157-163]`  
  → `172.16.1.157, 172.16.1.158, 172.16.1.159, 172.16.1.160, 172.16.1.161, 172.16.1.162, 172.16.1.163`
  
- `job-[001-005]`  
  → `job-001, job-002, job-003, job-004, job-005`
  
- `[1-3]`  
  → `1, 2, 3`

**Padding rules:**
- `[001-010]`: first number has 3 digits → all outputs padded to 3 digits
- `[1-10]`: first number has 1 digit → no padding
- `[01-100]`: first number has 2 digits → outputs padded to 2 digits until the number naturally exceeds that width (01, 02, ..., 99, 100)

### 2. Explicit List Syntax

**Format:** `prefix[item,item,item]suffix`

Expands to exactly the items listed. Items are used as-is with no numeric interpretation.

**Examples:**
- `job-[001,002,004,005]`  
  → `job-001, job-002, job-004, job-005`
  
- `host-[a,b,c]`  
  → `host-a, host-b, host-c`

### 3. Mixed Ranges and Lists

**Format:** `prefix[range,list,range,...]suffix`

Each comma-separated item inside `[...]` can be either a range (`start-end`) or a literal value.

**Examples:**
- `172.16.1.[157-159,161-163]`  
  → `172.16.1.157, 172.16.1.158, 172.16.1.159, 172.16.1.161, 172.16.1.162, 172.16.1.163`
  
- `job-[1-3,5,7-9]`  
  → `job-1, job-2, job-3, job-5, job-7, job-8, job-9`

### 4. Multiple Patterns

**Format:** `pattern,pattern,pattern`

Top-level commas (outside brackets) separate independent patterns or literals.

**Examples:**
- `static-id,job-[1-3],host-[a,b]`  
  → `static-id, job-1, job-2, job-3, host-a, host-b`

## Architecture

### New Module: `devrun/utils/pattern_expansion.py`

**Public API:**

```python
def expand_patterns(pattern_str: str) -> list[str]:
    """Expand job ID patterns into a flat list of job IDs.
    
    Supports:
      - Ranges: prefix[1-5]suffix → prefix1suffix, prefix2suffix, ...
      - Lists: prefix[a,b,c]suffix → prefixa, prefixb, prefixc
      - Mixed: prefix[1-3,5,7-9]suffix
      - Multiple: pattern1,pattern2,pattern3
    
    Args:
        pattern_str: Input string with optional bracket patterns
    
    Returns:
        List of expanded job IDs (strings)
    
    Raises:
        ValueError: Invalid syntax (unclosed brackets, start > end, non-numeric range)
    
    Examples:
        >>> expand_patterns("172.16.1.[157-163]")
        ['172.16.1.157', '172.16.1.158', ..., '172.16.1.163']
        
        >>> expand_patterns("job-[001-003,005]")
        ['job-001', 'job-002', 'job-003', 'job-005']
    """
```

**Internal Functions:**

```python
def _expand_single_pattern(pattern: str) -> list[str]:
    """Expand one pattern (may contain one [...] bracket).
    
    Returns list of expanded strings. No brackets → returns [pattern].
    """

def _expand_bracket_content(content: str, prefix: str, suffix: str) -> list[str]:
    """Parse content inside [...] and generate expanded values.
    
    Splits on comma, identifies ranges vs literals, expands each,
    then combines with prefix/suffix.
    """

def _expand_range(start_str: str, end_str: str) -> list[str]:
    """Expand numeric range with auto-detected padding.
    
    Args:
        start_str: Start number as string (e.g., "001")
        end_str: End number as string (e.g., "010")
    
    Returns:
        List of formatted strings with appropriate padding
    
    Raises:
        ValueError: start > end or non-numeric input
    """
```

### Integration Point

**File:** `devrun/tasks/swe_bench_agentic.py`  
**Location:** Lines 286-288 in `prepare_multi()` method

**Current code:**
```python
job_ids = params.get("job_ids")
if job_ids:
    instances = [{"JOB_ID": jid.strip()} for jid in str(job_ids).split(",")]
```

**Updated code:**
```python
from devrun.utils.pattern_expansion import expand_patterns

job_ids = params.get("job_ids")
if job_ids:
    expanded = expand_patterns(str(job_ids))
    instances = [{"JOB_ID": jid.strip()} for jid in expanded]
```

## Implementation Details

### Parsing Algorithm

1. **Top-level comma split:**
   - Split input string on commas that are NOT inside brackets
   - Use a state machine tracking bracket depth
   - Example: `a,b[1,2],c` → `["a", "b[1,2]", "c"]`

2. **Per-item expansion:**
   - If no brackets: return item as-is (literal)
   - If has brackets:
     - Extract prefix (before `[`), content (inside `[...]`), suffix (after `]`)
     - Parse bracket content: split on comma
     - For each segment:
       - Contains hyphen → range expansion
       - No hyphen → literal value
     - Combine: `prefix + expanded_value + suffix`

3. **Range expansion:**
   ```python
   def _expand_range(start_str: str, end_str: str) -> list[str]:
       start_int = int(start_str)
       end_int = int(end_str)
       
       if start_int > end_int:
           raise ValueError(f"Invalid range: start {start_int} > end {end_int}")
       
       # Detect padding from start_str
       pad_width = len(start_str) if start_str[0] == '0' else 0
       
       return [str(i).zfill(pad_width) for i in range(start_int, end_int + 1)]
   ```

### Error Handling

**Invalid syntax → `ValueError` with clear message:**

- `[10-5]` → `"Invalid range: start 10 > end 5"`
- `[abc-xyz]` → `"Non-numeric range: abc-xyz"`
- `[1-3` → `"Unclosed bracket in pattern: [1-3"`
- `[[1-3]]` → `"Nested brackets not supported: [[1-3]]"`
- `[]` → `"Empty bracket in pattern"`

### Edge Cases

| Input | Output | Notes |
|-------|--------|-------|
| Empty string | `[]` | No patterns to expand |
| Just literals `a,b,c` | `["a", "b", "c"]` | No brackets, pass through |
| Single item range `[5-5]` | `["5"]` | Valid, returns one item |
| Whitespace `[ 1 - 3 ]` | `["1", "2", "3"]` | Strip whitespace from segments |
| Leading zeros in list `[001,abc]` | `["001", "abc"]` | Keep as-is, no interpretation |
| Mixed padding `[1-3,001-003]` | `["1", "2", "3", "001", "002", "003"]` | Each range uses its own padding |

## Testing Strategy

### Unit Tests (`tests/test_pattern_expansion.py`)

**Basic range expansion:**
```python
def test_simple_range():
    assert expand_patterns("[1-5]") == ["1", "2", "3", "4", "5"]

def test_padded_range():
    assert expand_patterns("[001-005]") == ["001", "002", "003", "004", "005"]
```

**Range with prefix/suffix:**
```python
def test_range_with_prefix():
    assert expand_patterns("job-[1-3]") == ["job-1", "job-2", "job-3"]

def test_ip_range():
    result = expand_patterns("172.16.1.[157-159]")
    assert result == ["172.16.1.157", "172.16.1.158", "172.16.1.159"]
```

**Explicit lists:**
```python
def test_explicit_list():
    assert expand_patterns("[a,b,c]") == ["a", "b", "c"]

def test_list_with_prefix():
    result = expand_patterns("job-[001,003,005]")
    assert result == ["job-001", "job-003", "job-005"]
```

**Mixed ranges and lists:**
```python
def test_mixed_range_and_list():
    result = expand_patterns("[1-3,5,7-9]")
    assert result == ["1", "2", "3", "5", "7", "8", "9"]

def test_ip_mixed():
    result = expand_patterns("172.16.1.[157-159,161-163]")
    assert len(result) == 6
    assert result[0] == "172.16.1.157"
    assert result[-1] == "172.16.1.163"
```

**Multiple patterns (top-level comma):**
```python
def test_multiple_literals():
    assert expand_patterns("a,b,c") == ["a", "b", "c"]

def test_mixed_literal_and_pattern():
    result = expand_patterns("literal,job-[1-2]")
    assert result == ["literal", "job-1", "job-2"]
```

**Error cases:**
```python
def test_invalid_range_start_greater_than_end():
    with pytest.raises(ValueError, match="Invalid range"):
        expand_patterns("[5-3]")

def test_non_numeric_range():
    with pytest.raises(ValueError, match="Non-numeric"):
        expand_patterns("[abc-xyz]")

def test_unclosed_bracket():
    with pytest.raises(ValueError, match="Unclosed bracket"):
        expand_patterns("[1-3")

def test_nested_brackets():
    with pytest.raises(ValueError, match="Nested brackets"):
        expand_patterns("[[1-3]]")

def test_empty_bracket():
    with pytest.raises(ValueError, match="Empty bracket"):
        expand_patterns("[]")
```

**Edge cases:**
```python
def test_empty_string():
    assert expand_patterns("") == []

def test_whitespace_handling():
    result = expand_patterns("[ 1 - 3 , 5 ]")
    assert result == ["1", "2", "3", "5"]

def test_single_item_range():
    assert expand_patterns("[5-5]") == ["5"]
```

### Integration Tests (`tests/test_swe_bench_agentic.py`)

```python
def test_job_ids_pattern_expansion():
    """job_ids with pattern should expand to multiple instances."""
    task = SWEBenchAgenticTask()
    specs = task.prepare_multi(_make_params(
        job_ids="job-[1-3]",
        array="000-002"
    ))
    assert len(specs) == 3
    # Verify each spec has correct JOB_ID env var
    assert "JOB_ID=job-1" in specs[0].command or specs[0].env.get("JOB_ID") == "job-1"
    assert "JOB_ID=job-2" in specs[1].command or specs[1].env.get("JOB_ID") == "job-2"
    assert "JOB_ID=job-3" in specs[2].command or specs[2].env.get("JOB_ID") == "job-3"

def test_job_ids_mixed_pattern():
    """job_ids with mixed ranges and lists."""
    task = SWEBenchAgenticTask()
    specs = task.prepare_multi(_make_params(
        job_ids="172.16.1.[157-159,161-163]",
        array="000-005"
    ))
    assert len(specs) == 6
```

### Manual Validation

1. Create a YAML config with pattern:
   ```yaml
   task: swe_bench_agentic
   executor: slurm
   params:
     job_ids: "job-[001-003]"
     array: "000-002"
     dataset: /data/test
     llm_config: /configs/test.json
   ```

2. Run: `devrun run swe_bench_agentic --config test.yaml`

3. Verify:
   - 3 jobs enqueued
   - Generated sbatch scripts have correct `JOB_ID` env vars
   - Array sharding works correctly (000 → job-001, 001 → job-002, 002 → job-003)

## Backward Compatibility

**Full backward compatibility maintained:**

- Existing `job_ids` configs without patterns continue to work unchanged
- Plain comma-separated lists: `"id1,id2,id3"` → no change in behavior
- If pattern expansion fails (invalid syntax), raise `ValueError` with clear message — fail fast

## Future Extensions (Out of Scope)

Potential future enhancements not included in this design:

- Bash-style `{...}` syntax (e.g., `{001..010}` with step `{1..10..2}`)
- Wildcard matching against database (`job-*` finds all jobs matching prefix)
- Pattern expansion in other parameters (`instances`, `array`, file paths)
- Nested brackets or complex expressions

These can be added later if needed, but are deliberately excluded to keep this feature simple and focused.

## Summary

This design adds convenient pattern expansion for the `job_ids` parameter in SWE-bench tasks, allowing users to write compact patterns like `172.16.1.[157-163]` instead of listing individual IDs. The implementation is self-contained in a reusable utility module, fully tested, and maintains complete backward compatibility with existing configs.
