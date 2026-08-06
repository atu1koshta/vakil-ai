"""Parser component package.

Layout (the convention for every pipeline-component package):
  base.py      — the interface (ABC) + shared errors/helpers
  <impl>.py    — one implementation per file
  __init__.py  — registry + factory; the ONLY thing the rest of the app imports

Adding a parser = new module implementing Parser + one _PARSERS entry +
`components.parser: <name>` in config.yaml.
"""

from ..config import ConfigError, get_config
from .base import Parser
from .docling import DoclingParser

_PARSERS: dict[str, type[Parser]] = {"docling": DoclingParser}

# Process-lifetime instances: parsers hold heavyweight models.
_instances: dict[str, Parser] = {}


def get_parser(name: str | None = None) -> Parser:
    name = name or get_config().components.parser
    if name not in _PARSERS:
        raise ConfigError(f"unknown parser '{name}'; available: {sorted(_PARSERS)}")
    if name not in _instances:
        _instances[name] = _PARSERS[name]()
    return _instances[name]


__all__ = ["Parser", "DoclingParser", "get_parser"]
