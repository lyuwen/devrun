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

    def test_range_with_prefix_and_suffix(self):
        """172.16.1.[157-159] expands IP addresses."""
        result = expand_patterns("172.16.1.[157-159]")
        assert result == ["172.16.1.157", "172.16.1.158", "172.16.1.159"]


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


class TestEdgeCases:
    def test_no_brackets(self):
        """Plain string returns as-is."""
        result = expand_patterns("job-123")
        assert result == ["job-123"]

    def test_empty_string(self):
        """Empty input returns empty list."""
        result = expand_patterns("")
        assert result == []

    def test_whitespace_only(self):
        """Whitespace-only input returns empty list."""
        result = expand_patterns("   ")
        assert result == []

    def test_unclosed_bracket(self):
        """Raises ValueError for unclosed bracket."""
        with pytest.raises(ValueError, match="Unclosed bracket"):
            expand_patterns("job-[1-5")

    def test_empty_bracket(self):
        """Raises ValueError for empty bracket."""
        with pytest.raises(ValueError, match="Empty bracket"):
            expand_patterns("job-[]")

    def test_nested_brackets(self):
        """Raises ValueError for nested brackets."""
        with pytest.raises(ValueError, match="Nested brackets not supported"):
            expand_patterns("job-[1-[5-10]]")


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
