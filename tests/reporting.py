"""A reporter that keeps what a step reported, for tests to read.

``CollectingReporter`` implements ``moseq2_common.report.Reporter``: every status
line, warning and progress bar a step sends it is kept in order, and
``request_stop`` makes ``stop_requested`` answer True, as a GUI's Stop button
does.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Bar:
    label: str
    total: int | None
    unit: str
    done: int = 0
    closed: bool = False

    def advance(self, n: int = 1) -> None:
        self.done += n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


@dataclass
class CollectingReporter:
    statuses: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    bars: list = field(default_factory=list)
    stop: bool = False

    def status(self, text: str) -> None:
        self.statuses.append(text)

    def warning(self, text: str) -> None:
        self.warnings.append(text)

    def progress(self, label: str, total: int | None = None, *, unit: str = "") -> Bar:
        bar = Bar(label, total, unit)
        self.bars.append(bar)
        return bar

    def stop_requested(self) -> bool:
        return self.stop

    def request_stop(self) -> None:
        self.stop = True
