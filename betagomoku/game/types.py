from __future__ import annotations

import enum
from typing import NamedTuple


class Player(enum.Enum):
    BLACK = 1
    WHITE = 2

    @property
    def other(self) -> Player:
        return Player.WHITE if self is Player.BLACK else Player.BLACK

    def __str__(self) -> str:
        return self.name.capitalize()


class Point(NamedTuple):
    row: int  # 1-indexed, 1 = bottom
    col: int  # 1-indexed, 1 = left
