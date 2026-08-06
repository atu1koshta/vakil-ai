"""MetadataExtractor interface: parsed Markdown -> document header fields.

Pipeline-level component (output shared by all profiles); a change here must
bump PIPELINE_VERSION.
"""

from abc import ABC, abstractmethod

from ..models import DocumentMetadata


class MetadataExtractor(ABC):
    @abstractmethod
    def extract(self, markdown: str, source_file: str) -> DocumentMetadata: ...
