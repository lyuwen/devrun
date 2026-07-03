import pytest
from devrun.utils.pattern_expansion import expand_patterns, _expand_range


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

    def test_ip_range_expansion(self):
        """10.0.0.[101-103] expands IP addresses."""
        result = expand_patterns("10.0.0.[101-103]")
        assert result == ["10.0.0.101", "10.0.0.102", "10.0.0.103"]


class TestExpandRangeHelper:
    def test_simple_range(self):
        """Basic unpadded range."""
        result = _expand_range("1", "5")
        assert result == ["1", "2", "3", "4", "5"]

    def test_zero_padded_range(self):
        """Zero-padding preserved."""
        result = _expand_range("001", "005")
        assert result == ["001", "002", "003", "004", "005"]

    def test_mixed_padding(self):
        """Uses max length for padding."""
        result = _expand_range("8", "012")
        assert result == ["008", "009", "010", "011", "012"]

    def test_start_greater_than_end(self):
        """Raises ValueError when start > end."""
        with pytest.raises(ValueError, match="Range start 10 > end 5"):
            _expand_range("10", "5")

    def test_single_value_range(self):
        """Start == end returns single value."""
        result = _expand_range("42", "42")
        assert result == ["42"]


class TestErrorCases:
    def test_invalid_range_start_greater_than_end(self):
        """[10-5] raises ValueError."""
        with pytest.raises(ValueError, match="Range start 10 > end 5"):
            expand_patterns("[10-5]")

    def test_non_numeric_range(self):
        """[abc-xyz] treats as literal (non-numeric ranges don't raise)."""
        result = expand_patterns("[abc-xyz]")
        assert result == ["abc-xyz"]

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
    def test_no_brackets(self):
        """Plain string returns as-is."""
        result = expand_patterns("job-123")
        assert result == ["job-123"]

    def test_empty_string(self):
        """Empty string returns empty list."""
        result = expand_patterns("")
        assert result == []

    def test_whitespace_only(self):
        """Whitespace-only string returns empty list."""
        result = expand_patterns("   ")
        assert result == []

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
        """[01-100] uses max(start_width, end_width) for padding."""
        result = expand_patterns("[01-100]")
        assert result[0] == "001"  # max(2, 3) = 3 digits
        assert result[98] == "099"
        assert result[99] == "100"


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

    def test_multiple_ranges_with_gap(self):
        """10.0.0.[101-103,111-113] expands IP ranges with gap."""
        result = expand_patterns("10.0.0.[101-103,111-113]")
        assert result == [
            "10.0.0.101", "10.0.0.102", "10.0.0.103",
            "10.0.0.111", "10.0.0.112", "10.0.0.113"
        ]


    def test_mixed_with_prefix_suffix(self):
        """job-[1-3,5,7-9]-suffix expands correctly."""
        result = expand_patterns("job-[1-3,5,7-9]-end")
        assert result == [
            "job-1-end", "job-2-end", "job-3-end", "job-5-end",
            "job-7-end", "job-8-end", "job-9-end"
        ]


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

    def test_multiple_patterns_with_commas(self):
        """10.0.0.[1-2],192.168.1.[10-11] expands both ranges."""
        result = expand_patterns("10.0.0.[1-2],192.168.1.[10-11]")
        assert result == ["10.0.0.1", "10.0.0.2", "192.168.1.10", "192.168.1.11"]

