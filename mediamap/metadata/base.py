from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

RecordT = TypeVar("RecordT")


@dataclass(frozen=True)
class MetadataQuery:
    title: str | None = None
    creator: str | None = None
    identifier: str | None = None


class MetadataProvider(ABC, Generic[RecordT]):
    """Common search contract for external metadata sources."""

    name: str

    @abstractmethod
    def search(self, query: MetadataQuery) -> list[RecordT] | None:
        """Search this provider; return None when the lookup could not complete."""
