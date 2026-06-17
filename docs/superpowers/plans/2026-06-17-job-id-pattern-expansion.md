# Job ID Pattern Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pattern expansion support for `job_ids` parameter in SWE-bench agentic tasks to handle patterns like `172.16.1.[157-163]` and `job-[1-3,5,7-9]`.

**Architecture:** Create a reusable `devrun/utils/pattern_expansion.py` module with a parser that handles bracket-based patterns (ranges and lists). Integrate into `SWEBenchAgenticTask.prepare_multi()` by replacing the simple comma-split logic with pattern expansion.

**Tech Stack:** Python 3.10+, pytest, regex

## Global Constraints

- Python 3.10+ required
- Follow existing devrun code style (type hints, docstrings)
- All public functions must have comprehensive docstrings
- Use stdlib only (no new dependencies)
- Maintain backward compatibility with existing `job_ids` usage

---

## File Structure

**New files:**
- `devrun/utils/pattern_expansion.py` — pattern expansion logic
- `tests/test_pattern_expansion.py` — unit tests for pattern expansion

**Modified files:**
- `devrun/tasks/swe_bench_agentic.py:286-288` — integrate pattern expansion into `prepare_multi()`
- `tests/test_swe_bench_agentic.py` — add integration tests

---

### Task 1: Create pattern expansion module with range support

**Files:**
- Create: `devrun/utils/pattern_expansion.py`
- Test: `tests/test_pattern_expansion.py`

**Interfaces:**
- Produces: `expand_patterns(pattern_str: str) -> list[str]`
- Produces: `_expand_range(start_str: str, end_str: str) -> list[str]`

- [ ] **Step 1: Write failing test for simple range**

Create `tests/test_pattern_expansion.py`:

```python
import pytest
from devrun.utils.pattern_expansion import expand_patterns


class TestSimpleRanges:
    def test_simple_numeric_range(self):
        """[1-5] expands to 1, 2, 3, 4, 5."""
        result = expand_patterns("[1-5]")
        assert result == ["1", "2", "3", "4", "5"]

    def test_padded_range(self):
        """[001-005] expands with zero-padding preserved."""
        result = expand_patterns("[001-005]")
        assert result == ["001", "002", "003", "004", "005"]

    def test_range_with_prefix(self):
        """job-[1-3] expands to job-1, job-2, job-3."""
        result = expand_patterns("job-[1-3]")
        assert result == ["job-1", "job-2", "job-3"]

    def test_range_with_prefix_and_suffix(self):
        """172.16.1.[157-159] expands IP addresses."""
        result = expand_patterns("172.16.1.[157-159]")
        assert result == ["172.16.1.157", "172.16.1.158", "172.16.1.159"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_pattern_expansion.py::TestSimpleRanges -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'devrun.utils.pattern_expansion'`

- [ ] **Step 3: Create module skeleton with _expand_range helper**

Create `devrun/utils/pattern_expansion.py`:

```python
"""Pattern expansion utilities for job IDs and similar identifiers.

Supports bracket-based patterns:
  - Ranges: prefix[1-5]suffix → prefix1suffix, prefix2suffix, ...
  - Lists: prefix[a,b,c]suffix → prefixa, prefixb, prefixc
  - Mixed: prefix[1-3,5,7-9]suffix
"""

from typing import Any


def expand_patterns(pattern_str: str) -> list[str]:
    """Expand job ID patterns into a flat list of values.
    
    Supports:
      - Ranges: prefix[1-5]suffix → prefix1suffix, prefix2suffix, ...
      - Lists: prefix[a,b,c]suffix → prefixa, prefixb, prefixc
      - Mixed: prefix[1-3,5,7-9]suffix
      - Multiple: pattern1,pattern2,pattern3
    
    Args:
        pattern_str: Input string with optional bracket patterns
    
    Returns:
        List of expanded values (strings)
    
    Raises:
        ValueError: Invalid syntax (unclosed brackets, start > end, non-numeric range)
    
    Examples:
        >>> expand_patterns("172.16.1.[157-163]")
        ['172.16.1.157', '172.16.1.158', ..., '172.16.1.163']
        
        >>> expand_patterns("job-[001-003,005]")
        ['job-001', 'job-002', 'job-003', 'job-005']
    """
    if not pattern_str or not pattern_str.strip():
        return []
    
    # For now, delegate to _expand_single_pattern
    # (will add top-level comma handling later)
    return _expand_single_pattern(pattern_str.strip())


def _expand_single_pattern(pattern: str) -> list[str]:
    """Expand one pattern (may contain one [...] bracket).
    
    Returns list of expanded strings. No brackets → returns [pattern].
    """
    if "[" not in pattern:
        return [pattern]
    
    # Find bracket positions
    start_idx = pattern.index("[")
    try:
        end_idx = pattern.index("]", start_idx)
    except ValueError:
        raise ValueError(f"Unclosed bracket in pattern: {pattern}")
    
    # Check for nested brackets
    if "[" in pattern[start_idx + 1:end_idx]:
        raise ValueError(f"Nested brackets not supported: {pattern}")
    
    prefix = pattern[:start_idx]
    content = pattern[start_idx + 1:end_idx]
    suffix = pattern[end_idx + 1:]
    
    if not content:
        raise ValueError(f"Empty bracket in pattern: {pattern}")
    
    return _expand_bracket_content(content, prefix, suffix)


def _expand_bracket_content(content: str, prefix: str, suffix: str) -> list[str]:
    """Parse content inside [...] and generate expanded values.
    
    Splits on comma, identifies ranges vs literals, expands each,
    then combines with prefix/suffix.
    """
    result = []
    segments = [seg.strip() for seg in content.split(",")]
    
    for seg in segments:
        if "-" in seg and seg.count("-") == 1:
            # Potential range
            parts = seg.split("-")
            start_str = parts[0].strip()
            end_str = parts[1].strip()
            
            # Check if both are numeric
            if start_str.isdigit() and end_str.isdigit():
                expanded = _expand_range(start_str, end_str)
                result.extend([f"{prefix}{val}{suffix}" for val in expanded])
            else:
                # Treat as literal (e.g., "abc-def")
                result.append(f"{prefix}{seg}{suffix}")
        else:
            # Literal value
            result.append(f"{prefix}{seg}{suffix}")
    
    return result


def _expand_range(start_str: str, end_str: str) -> list[str]:
    """Expand numeric range with auto-detected padding.
    
    Args:
        start_str: Start number as string (e.g., "001")
        end_str: End number as string (e.g., "010")
    
    Returns:
        List of formatted strings with appropriate padding
    
    Raises:
        ValueError: start > end or non-numeric input
    
    Examples:
        >>> _expand_range("1", "5")
        ['1', '2', '3', '4', '5']
        
        >>> _expand_range("001", "005")
        ['001', '002', '003', '004', '005']
    """
    try:
        start_int = int(start_str)
        end_int = int(end_str)
    except ValueError:
        raise ValueError(f"Non-numeric range: {start_str}-{end_str}")
    
    if start_int > end_int:
        raise ValueError(f"Invalid range: start {start_int} > end {end_int}")
    
    # Detect padding from start_str
    pad_width = len(start_str) if start_str.startswith("0") else 0
    
    return [str(i).zfill(pad_width) for i in range(start_int, end_int + 1)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_pattern_expansion.py::TestSimpleRanges -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add devrun/utils/pattern_expansion.py tests/test_pattern_expansion.py
git commit -m "feat(utils): add pattern expansion with range support"
```

---

### Task 2: Add explicit list and mixed pattern support

**Files:**
- Modify: `devrun/utils/pattern_expansion.py`
- Modify: `tests/test_pattern_expansion.py`

**Interfaces:**
- Consumes: `expand_patterns(pattern_str: str) -> list[str]` from Task 1
- Produces: Enhanced `expand_patterns` that handles lists and mixed patterns

- [ ] **Step 1: Write failing tests for lists and mixed patterns**

Add to `tests/test_pattern_expansion.py`:

```python
class TestExplicitLists:
    def test_simple_list(self):
        """[a,b,c] expands to a, b, c."""
        result = expand_patterns("[a,b,c]")
        assert result == ["a", "b", "c"]

    def test_numeric_list(self):
        """[001,003,005] expands to non-sequential IDs."""
        result = expand_patterns("[001,003,005]")
        assert result == ["001", "003", "005"]

    def test_list_with_prefix(self):
        """job-[001,002,004,005] expands correctly."""
        result = expand_patterns("job-[001,002,004,005]")
        assert result == ["job-001", "job-002", "job-004", "job-005"]


class TestMixedPatterns:
    def test_mixed_range_and_list(self):
        """[1-3,5,7-9] expands with gaps."""
        result = expand_patterns("[1-3,5,7-9]")
        assert result == ["1", "2", "3", "5", "7", "8", "9"]

    def test_ip_mixed_pattern(self):
        """172.16.1.[157-159,161-163] expands IP ranges with gap."""
        result = expand_patterns("172.16.1.[157-159,161-163]")
        assert result == [
            "172.16.1.157", "172.16.1.158", "172.16.1.159",
            "172.16.1.161", "172.16.1.162", "172.16.1.163"
        ]

    def test_mixed_with_prefix_suffix(self):
        """job-[1-3,5,7-9]-suffix expands correctly."""
        result = expand_patterns("job-[1-3,5,7-9]-end")
        assert result == [
            "job-1-end", "job-2-end", "job-3-end", "job-5-end",
            "job-7-end", "job-8-end", "job-9-end"
        ]
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/test_pattern_expansion.py::TestExplicitLists -v
python -m pytest tests/test_pattern_expansion.py::TestMixedPatterns -v
```

Expected: All 6 tests PASS (implementation already handles these cases)

- [ ] **Step 3: Commit**

```bash
git add tests/test_pattern_expansion.py
git commit -m "test(pattern_expansion): add list and mixed pattern tests"
```

---

### Task 3: Add top-level comma splitting support

**Files:**
- Modify: `devrun/utils/pattern_expansion.py`
- Modify: `tests/test_pattern_expansion.py`

**Interfaces:**
- Consumes: `expand_patterns(pattern_str: str) -> list[str]` from Task 2
- Produces: Enhanced `expand_patterns` that handles multiple comma-separated patterns

- [ ] **Step 1: Write failing test for top-level comma splitting**

Add to `tests/test_pattern_expansion.py`:

```python
class TestMultiplePatterns:
    def test_multiple_literals(self):
        """a,b,c expands to three items."""
        result = expand_patterns("a,b,c")
        assert result == ["a", "b", "c"]

    def test_mixed_literal_and_pattern(self):
        """literal,job-[1-2] expands correctly."""
        result = expand_patterns("literal,job-[1-2]")
        assert result == ["literal", "job-1", "job-2"]

    def test_multiple_patterns(self):
        """static-id,job-[1-3],host-[a,b] expands all patterns."""
        result = expand_patterns("static-id,job-[1-3],host-[a,b]")
        assert result == ["static-id", "job-1", "job-2", "job-3", "host-a", "host-b"]

    def test_ip_ranges_multiple(self):
        """172.16.1.[1-2],192.168.1.[10-11] expands both ranges."""
        result = expand_patterns("172.16.1.[1-2],192.168.1.[10-11]")
        assert result == ["172.16.1.1", "172.16.1.2", "192.168.1.10", "192.168.1.11"]
```

- [ ] **Step 2: Run tests to verify some fail**

```bash
python -m pytest tests/test_pattern_expansion.py::TestMultiplePatterns -v
```

Expected: Tests likely FAIL because `expand_patterns` doesn't handle top-level commas yet

- [ ] **Step 3: Implement top-level comma splitting**

Modify `devrun/utils/pattern_expansion.py`, update `expand_patterns` function:

```python
def expand_patterns(pattern_str: str) -> list[str]:
    """Expand job ID patterns into a flat list of values.
    
    Supports:
      - Ranges: prefix[1-5]suffix → prefix1suffix, prefix2suffix, ...
      - Lists: prefix[a,b,c]suffix → prefixa, prefixb, prefixc
      - Mixed: prefix[1-3,5,7-9]suffix
      - Multiple: pattern1,pattern2,pattern3
    
    Args:
        pattern_str: Input string with optional bracket patterns
    
    Returns:
        List of expanded values (strings)
    
    Raises:
        ValueError: Invalid syntax (unclosed brackets, start > end, non-numeric range)
    
    Examples:
        >>> expand_patterns("172.16.1.[157-163]")
        ['172.16.1.157', '172.16.1.158', ..., '172.16.1.163']
        
        >>> expand_patterns("job-[001-003,005]")
        ['job-001', 'job-002', 'job-003', 'job-005']
    """
    if not pattern_str or not pattern_str.strip():
        return []
    
    # Split on top-level commas (not inside brackets)
    patterns = _split_top_level_commas(pattern_str.strip())
    
    # Expand each pattern independently
    result = []
    for pattern in patterns:
        result.extend(_expand_single_pattern(pattern.strip()))
    
    return result


def _split_top_level_commas(s: str) -> list[str]:
    """Split string on commas that are NOT inside brackets.
    
    Uses a simple state machine tracking bracket depth.
    
    Examples:
        >>> _split_top_level_commas("a,b,c")
        ['a', 'b', 'c']
        
        >>> _split_top_level_commas("a,b[1,2],c")
        ['a', 'b[1,2]', 'c']
    """
    result = []
    current = []
    depth = 0
    
    for char in s:
        if char == "[":
            depth += 1
            current.append(char)
        elif char == "]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            result.append("".join(current))
            current = []
        else:
            current.append(char)
    
    if current:
        result.append("".join(current))
    
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_pattern_expansion.py::TestMultiplePatterns -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add devrun/utils/pattern_expansion.py tests/test_pattern_expansion.py
git commit -m "feat(pattern_expansion): add top-level comma splitting"
```

---

### Task 4: Add comprehensive error handling tests

**Files:**
- Modify: `tests/test_pattern_expansion.py`

**Interfaces:**
- Consumes: `expand_patterns(pattern_str: str) -> list[str]` from Task 3
- Produces: Comprehensive test coverage for error cases

- [ ] **Step 1: Write tests for error cases**

Add to `tests/test_pattern_expansion.py`:

```python
class TestErrorCases:
    def test_invalid_range_start_greater_than_end(self):
        """[10-5] raises ValueError."""
        with pytest.raises(ValueError, match="Invalid range"):
            expand_patterns("[10-5]")

    def test_non_numeric_range(self):
        """[abc-xyz] raises ValueError."""
        with pytest.raises(ValueError, match="Non-numeric range"):
            expand_patterns("[abc-xyz]")

    def test_unclosed_bracket(self):
        """[1-3 raises ValueError."""
        with pytest.raises(ValueError, match="Unclosed bracket"):
            expand_patterns("[1-3")

    def test_nested_brackets(self):
        """[[1-3]] raises ValueError."""
        with pytest.raises(ValueError, match="Nested brackets"):
            expand_patterns("[[1-3]]")

    def test_empty_bracket(self):
        """[] raises ValueError."""
        with pytest.raises(ValueError, match="Empty bracket"):
            expand_patterns("[]")


class TestEdgeCases:
    def test_empty_string(self):
        """Empty string returns empty list."""
        assert expand_patterns("") == []

    def test_whitespace_only(self):
        """Whitespace-only string returns empty list."""
        assert expand_patterns("   ") == []

    def test_whitespace_in_bracket(self):
        """[ 1 - 3 , 5 ] handles whitespace correctly."""
        result = expand_patterns("[ 1 - 3 , 5 ]")
        assert result == ["1", "2", "3", "5"]

    def test_single_item_range(self):
        """[5-5] returns single item."""
        assert expand_patterns("[5-5]") == ["5"]

    def test_literal_with_hyphen(self):
        """[abc-def,ghi] treats abc-def as literal (non-numeric)."""
        result = expand_patterns("[abc-def,ghi]")
        assert result == ["abc-def", "ghi"]

    def test_mixed_padding(self):
        """[1-3,001-003] uses different padding per range."""
        result = expand_patterns("[1-3,001-003]")
        assert result == ["1", "2", "3", "001", "002", "003"]

    def test_padding_overflow(self):
        """[01-100] pads until natural width exceeds pad_width."""
        result = expand_patterns("[01-100]")
        assert result[0] == "01"
        assert result[98] == "99"
        assert result[99] == "100"  # Natural width exceeds pad_width
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/test_pattern_expansion.py::TestErrorCases -v
python -m pytest tests/test_pattern_expansion.py::TestEdgeCases -v
```

Expected: All 16 tests PASS

- [ ] **Step 3: Run full test suite for pattern_expansion**

```bash
python -m pytest tests/test_pattern_expansion.py -v
```

Expected: All tests PASS (should be ~30 tests total)

- [ ] **Step 4: Commit**

```bash
git add tests/test_pattern_expansion.py
git commit -m "test(pattern_expansion): add error and edge case coverage"
```

---

### Task 5: Integrate pattern expansion into SWEBenchAgenticTask

**Files:**
- Modify: `devrun/tasks/swe_bench_agentic.py:286-288`
- Modify: `tests/test_swe_bench_agentic.py`

**Interfaces:**
- Consumes: `expand_patterns(pattern_str: str) -> list[str]` from Task 4
- Produces: Enhanced `SWEBenchAgenticTask.prepare_multi()` with pattern support

- [ ] **Step 1: Write failing integration test**

Add to `tests/test_swe_bench_agentic.py`:

```python
def test_job_ids_pattern_expansion():
    """job_ids with pattern should expand to multiple instances."""
    task = SWEBenchAgenticTask()
    specs = task.prepare_multi(_make_params(
        job_ids="job-[1-3]",
        array="000-002"
    ))
    assert len(specs) == 3
    # Check that JOB_ID env var is set correctly in each spec
    assert "JOB_ID" in specs[0].env
    assert specs[0].env["JOB_ID"] == "job-1"
    assert specs[1].env["JOB_ID"] == "job-2"
    assert specs[2].env["JOB_ID"] == "job-3"


def test_job_ids_mixed_pattern():
    """job_ids with mixed ranges and lists."""
    task = SWEBenchAgenticTask()
    specs = task.prepare_multi(_make_params(
        job_ids="172.16.1.[157-159,161-163]",
        array="000-005"
    ))
    assert len(specs) == 6
    expected_ids = [
        "172.16.1.157", "172.16.1.158", "172.16.1.159",
        "172.16.1.161", "172.16.1.162", "172.16.1.163"
    ]
    for i, spec in enumerate(specs):
        assert spec.env["JOB_ID"] == expected_ids[i]


def test_job_ids_backward_compatibility():
    """Plain comma-separated job_ids still works."""
    task = SWEBenchAgenticTask()
    specs = task.prepare_multi(_make_params(
        job_ids="id1,id2,id3",
        array="000-002"
    ))
    assert len(specs) == 3
    assert specs[0].env["JOB_ID"] == "id1"
    assert specs[1].env["JOB_ID"] == "id2"
    assert specs[2].env["JOB_ID"] == "id3"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_swe_bench_agentic.py::test_job_ids_pattern_expansion -v
python -m pytest tests/test_swe_bench_agentic.py::test_job_ids_mixed_pattern -v
python -m pytest tests/test_swe_bench_agentic.py::test_job_ids_backward_compatibility -v
```

Expected: FAIL (pattern expansion not integrated yet)

- [ ] **Step 3: Integrate pattern expansion into prepare_multi**

Modify `devrun/tasks/swe_bench_agentic.py`, find the `prepare_multi` method around line 286:

```python
# Add import at top of file
from devrun.utils.pattern_expansion import expand_patterns

# Then in prepare_multi method, replace lines 286-288:
# OLD CODE:
# job_ids = params.get("job_ids")
# if job_ids:
#     instances = [{"JOB_ID": jid.strip()} for jid in str(job_ids).split(",")]

# NEW CODE:
job_ids = params.get("job_ids")
if job_ids:
    expanded = expand_patterns(str(job_ids))
    instances = [{"JOB_ID": jid.strip()} for jid in expanded]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_swe_bench_agentic.py::test_job_ids_pattern_expansion -v
python -m pytest tests/test_swe_bench_agentic.py::test_job_ids_mixed_pattern -v
python -m pytest tests/test_swe_bench_agentic.py::test_job_ids_backward_compatibility -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Run full swe_bench_agentic test suite**

```bash
python -m pytest tests/test_swe_bench_agentic.py -v
```

Expected: All existing tests still PASS (backward compatibility maintained)

- [ ] **Step 6: Commit**

```bash
git add devrun/tasks/swe_bench_agentic.py tests/test_swe_bench_agentic.py
git commit -m "feat(swe_bench_agentic): integrate pattern expansion for job_ids"
```

---

### Task 6: Run full test suite and update documentation

**Files:**
- Modify: `docs/features/swe_bench_script_customization.md` (add pattern syntax section)

**Interfaces:**
- Consumes: All components from Tasks 1-5
- Produces: Documentation and verified end-to-end functionality

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS (no regressions)

- [ ] **Step 2: Update documentation with pattern syntax**

Modify `docs/features/swe_bench_script_customization.md`, add new section after the `job_ids` shorthand section:

```markdown
## Pattern Expansion for job_ids

The `job_ids` parameter supports pattern expansion for convenient specification of multiple job IDs.

### Range Syntax

Use `[start-end]` to expand numeric ranges:

```yaml
params:
  job_ids: "172.16.1.[157-163]"
  # Expands to: 172.16.1.157, 172.16.1.158, ..., 172.16.1.163
```

Padding is auto-detected from the first number:
```yaml
params:
  job_ids: "job-[001-005]"
  # Expands to: job-001, job-002, job-003, job-004, job-005
```

### Explicit List Syntax

Use `[item,item,item]` for non-sequential values:

```yaml
params:
  job_ids: "job-[001,002,004,005]"
  # Expands to: job-001, job-002, job-004, job-005
```

### Mixed Ranges and Lists

Combine ranges and lists within a single bracket:

```yaml
params:
  job_ids: "172.16.1.[157-159,161-163]"
  # Expands to: 172.16.1.157, 172.16.1.158, 172.16.1.159,
  #             172.16.1.161, 172.16.1.162, 172.16.1.163
```

### Multiple Patterns

Separate multiple patterns with commas:

```yaml
params:
  job_ids: "static-id,job-[1-3],host-[a,b]"
  # Expands to: static-id, job-1, job-2, job-3, host-a, host-b
```

### Backward Compatibility

Plain comma-separated lists continue to work unchanged:

```yaml
params:
  job_ids: "id1,id2,id3"
  # Still works exactly as before
```
```

- [ ] **Step 3: Commit documentation**

```bash
git add docs/features/swe_bench_script_customization.md
git commit -m "docs: add pattern expansion syntax for job_ids"
```

- [ ] **Step 4: Manual verification (optional)**

Create a test YAML config and verify pattern expansion works:

```yaml
# /tmp/test_pattern.yaml
task: swe_bench_agentic
executor: slurm
params:
  job_ids: "test-[001-003]"
  array: "000-002"
  dataset: /data/test
  llm_config: /configs/test.json
```

Then run:
```bash
devrun run swe_bench_agentic --config /tmp/test_pattern.yaml --dry-run
```

Verify output shows 3 jobs with JOB_ID=test-001, test-002, test-003

---

## Plan Complete

All tasks implement the job ID pattern expansion feature as specified in the design document. The implementation:

- ✅ Supports numeric ranges with auto-detected padding
- ✅ Supports explicit lists
- ✅ Supports mixed ranges and lists within brackets
- ✅ Supports multiple comma-separated patterns
- ✅ Maintains backward compatibility
- ✅ Comprehensive test coverage (~35 tests)
- ✅ Clear error messages for invalid syntax
- ✅ Reusable utility module for future use
- ✅ Documentation updated

Each task is independently testable with clear success criteria and frequent commits following TDD principles.
