"""Parser interface: PDF -> structured Markdown.

Parsing is PIPELINE-level (its output in output/ is shared by all profiles) —
a parser change must bump PIPELINE_VERSION, not a profile.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class Parser(ABC):
    @abstractmethod
    def parse(self, pdf_path: Path) -> str: ...
