from __future__ import annotations

import abc

from betagomoku.game.board import GomokuGameState
from betagomoku.game.types import Point


class Agent(abc.ABC):
    @abc.abstractmethod
    def select_move(self, game_state: GomokuGameState) -> Point:
        """Return the point where this agent wants to play."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
