"""日志与计时工具。"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

_LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根 logger。"""
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
    )
    # 降低第三方库噪音
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class Stopwatch:
    """
    累计计时器。

    用法:
        sw = Stopwatch()
        with sw.section("pose"):
            ...
        print(sw.report())
    """

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._stack: list[tuple[str, float]] = []

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stages[name] = self.stages.get(name, 0.0) + elapsed

    def report(self, unit: str = "ms") -> str:
        total = sum(self.stages.values())
        scale = 1000.0 if unit == "ms" else 1.0
        lines = [f"总耗时: {total*scale:.2f} {unit}"]
        for k, v in self.stages.items():
            pct = (v / total * 100) if total > 0 else 0.0
            lines.append(f"  {k:<22s} {v*scale:9.2f} {unit}  ({pct:5.1f}%)")
        return "\n".join(lines)


def format_duration(seconds: float) -> str:
    """秒 → MM:SS.s"""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"
