from abc import ABC, abstractmethod
from pathlib import Path
from numpy import ndarray


class Method(ABC):
    @abstractmethod
    def assert_(base: Path | None, gt: Path | None, fn: list[Path] | None) -> None: ...

    @abstractmethod
    def process(
        base: ndarray | None, gt: ndarray | None, fn: list[ndarray] | None
    ) -> ndarray: ...


class DashboardMethod(ABC):
    @abstractmethod
    def assert_(base: Path | None, gt: Path | None, fn: list[Path] | None) -> None: ...

    @abstractmethod
    def process(
        base: ndarray | None, gt: ndarray | None, fn: list[ndarray] | None
    ) -> dict[str, object]: ...
