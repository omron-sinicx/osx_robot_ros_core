"""Lightweight TimeStep type — replaces dm_env.TimeStep/StepType."""

from typing import Any, NamedTuple

STEP_FIRST = 0
STEP_MID = 1
STEP_LAST = 2


class TimeStep(NamedTuple):
    step_type: int
    reward: Any
    discount: Any
    observation: Any

    def first(self) -> bool:
        return self.step_type == STEP_FIRST

    def mid(self) -> bool:
        return self.step_type == STEP_MID

    def last(self) -> bool:
        return self.step_type == STEP_LAST
