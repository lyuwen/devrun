"""Pattern expansion utilities for job IDs and similar identifiers.

Supports bracket-based patterns:
  - Ranges: prefix[1-5]suffix → prefix1suffix, prefix2suffix, ...
  - Lists: prefix[a,b,c]suffix → prefixa, prefixb, prefixc
  - Mixed: prefix[1-3,5,7-9]suffix
"""


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
    start = int(start_str)
    end = int(end_str)

    if start > end:
        raise ValueError(f"Range start {start} > end {end}")

    # Detect padding: use the max length of the two strings
    width = max(len(start_str), len(end_str))

    return [str(i).zfill(width) for i in range(start, end + 1)]
