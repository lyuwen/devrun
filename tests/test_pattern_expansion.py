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
