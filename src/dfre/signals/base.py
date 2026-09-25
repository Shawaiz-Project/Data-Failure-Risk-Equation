"""Base class for all signal extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Signal(ABC):
    """Abstract signal extractor.

    Subclass and implement :meth:`compute`. Every signal returns a float in [0, 1].
    """

    name: str = "signal"

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> float:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return self.compute(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
