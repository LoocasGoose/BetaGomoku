from __future__ import annotations

import random

from betagomoku.game.board import GomokuGameState
from betagomoku.game.types import Point

from .base import Agent


class RandomAgent(Agent):
    def select_move(self, game_state: GomokuGameState) -> Point:
        moves = game_state.legal_moves()
        assert moves, "No legal moves available"
        return random.choice(moves)
