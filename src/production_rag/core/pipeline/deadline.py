import time
from collections.abc import Callable


class Deadline:
    """A request-wide time budget; every stage spends from the same clock."""

    def __init__(self, budget_ms: int, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._end = clock() + budget_ms / 1000

    def remaining_s(self) -> float:
        return max(0.0, self._end - self._clock())

    def remaining_ms(self) -> int:
        return round(self.remaining_s() * 1000)

    def expired(self) -> bool:
        return self.remaining_s() <= 0
