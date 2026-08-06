"""VectorIndex interface: persistent vector index for ONE profile.

PROFILE-level component: vectors from different embedding models or chunking
configs live in unrelated spaces, so every implementation binds a single
profile and (for file-backed stores) verifies a config stamp on open.
"""

from abc import ABC, abstractmethod

from ..config import Profile


class IndexConfigMismatch(RuntimeError):
    pass


class VectorIndex(ABC):
    profile: Profile  # implementations must bind the owning profile

    @abstractmethod
    def existing_hashes(self, doc_id: str) -> dict[str, str]: ...

    @abstractmethod
    def upsert_chunk(
        self,
        *,
        key: str,
        doc_id: str,
        chunk_id: str,
        section: str,
        case_title: str,
        text: str,
        content_hash: str,
        vector: list[float],
    ) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], k: int = 5) -> list[dict]: ...

    @abstractmethod
    def count_chunks(self) -> int: ...

    @abstractmethod
    def count_for_doc(self, doc_id: str) -> int: ...

    @abstractmethod
    def commit(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "VectorIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
