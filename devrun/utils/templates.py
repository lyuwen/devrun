"""Jinja2 template rendering utility for devrun task plugins."""
from __future__ import annotations

import json
import shlex
from pathlib import Path

import jinja2
from jinja2 import Environment, PackageLoader, FileSystemLoader


def _get_jinja_env() -> Environment:
    """Return a configured Jinja2 Environment with devrun filters."""
    env = Environment(
        loader=PackageLoader("devrun", "templates"),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    env.filters["shell_quote"] = shlex.quote
    env.filters["tojson"] = json.dumps
    return env


def render_template(template_name: str, **kwargs: object) -> str:
    """Load a Jinja2 template from ``devrun/templates/`` and render it.

    If *template_name* is an absolute path, load it from the filesystem
    directly. Otherwise, load from the package's ``templates/`` directory.

    All user-supplied values interpolated into shell contexts should use
    the ``shell_quote`` filter: ``{{ value | shell_quote }}``.
    """
    # Check if template_name is an absolute path
    template_path = Path(template_name)
    if template_path.is_absolute():
        # Load from filesystem
        env = Environment(
            loader=FileSystemLoader(template_path.parent),
            keep_trailing_newline=True,
            undefined=jinja2.StrictUndefined,
        )
        env.filters["shell_quote"] = shlex.quote
        env.filters["tojson"] = json.dumps
        template = env.get_template(template_path.name)
    else:
        # Load from package templates directory
        env = _get_jinja_env()
        template = env.get_template(template_name)

    return template.render(**kwargs)
