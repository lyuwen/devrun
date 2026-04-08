"""Jinja2 template rendering utility for devrun task plugins."""
from __future__ import annotations

import shlex

import jinja2
from jinja2 import Environment, PackageLoader


def _get_jinja_env() -> Environment:
    """Return a configured Jinja2 Environment with devrun filters."""
    env = Environment(
        loader=PackageLoader("devrun", "templates"),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    env.filters["shell_quote"] = shlex.quote
    return env


def render_template(template_name: str, **kwargs: object) -> str:
    """Load a Jinja2 template from ``devrun/templates/`` and render it.

    All user-supplied values interpolated into shell contexts should use
    the ``shell_quote`` filter: ``{{ value | shell_quote }}``.
    """
    env = _get_jinja_env()
    template = env.get_template(template_name)
    return template.render(**kwargs)
