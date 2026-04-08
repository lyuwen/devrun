"""Unit tests for Jinja2 template rendering utility."""
from __future__ import annotations

import jinja2
import pytest
from devrun.utils.templates import render_template, _get_jinja_env


class TestRenderTemplate:
    def test_render_nonexistent_template_raises(self):
        with pytest.raises(Exception):
            render_template("nonexistent.j2")

    def test_shell_quote_filter_available(self):
        """The shell_quote filter must be registered."""
        env = _get_jinja_env()
        assert "shell_quote" in env.filters

    def test_shell_quote_filter_quotes_spaces(self):
        env = _get_jinja_env()
        tmpl = env.from_string("{{ val | shell_quote }}")
        result = tmpl.render(val="hello world")
        assert result == "'hello world'"

    def test_shell_quote_filter_quotes_special_chars(self):
        env = _get_jinja_env()
        tmpl = env.from_string("{{ val | shell_quote }}")
        result = tmpl.render(val="it's a test")
        assert "'" in result  # shlex.quote wraps in quotes

    def test_strict_undefined_raises_on_missing(self):
        """StrictUndefined should raise when a variable is missing."""
        env = _get_jinja_env()
        tmpl = env.from_string("Hello {{ missing_var }}")
        with pytest.raises(jinja2.UndefinedError):
            tmpl.render()
