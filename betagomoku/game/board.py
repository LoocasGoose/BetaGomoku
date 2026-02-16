from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import Player, Point

BOARD_SIZE = 15
WIN_LENGTH = 5

# Column labels: A-O (skipping no letters for 15x15)
COL_LABELS = "ABCDEFGHIJKLMNO"


def parse_coordinate(text: str) -> Optional[Point]:
    """Parse a coordinate string like 'E5' or 'H12' into a Point.

    Column is a letter A-O, row is a number 1-15.
    Returns None if the string is invalid.
    """
    text = text.strip().upper()
    if len(text) < 2 or len(text) > 3:
        return None
    col_char = text[0]
    row_str = text[1:]
    if col_char not in COL_LABELS:
        return None
    try:
        row = int(row_str)
    except ValueError:
        return None
    if not (1 <= row <= BOARD_SIZE):
        return None
    col = COL_LABELS.index(col_char) + 1
    return Point(row, col)


def format_point(point: Point) -> str:
    """Format a Point as a coordinate string like 'E5'."""
    return f"{COL_LABELS[point.col - 1]}{point.row}"


@dataclass
class Move:
    point: Point
    player: Player

    def __str__(self) -> str:
        return f"{self.player}: {format_point(self.point)}"


class Board:
    """15x15 Gomoku board. Tracks stone placement."""

    def __init__(self) -> None:
        self._grid: dict[Point, Player] = {}

    def place(self, point: Point, player: Player) -> None:
        assert self.is_empty(point), f"{format_point(point)} is occupied"
        self._grid[point] = player

    def remove(self, point: Point) -> None:
        del self._grid[point]

    def get(self, point: Point) -> Optional[Player]:
        return self._grid.get(point)

    def is_empty(self, point: Point) -> bool:
        return point not in self._grid

    def is_on_grid(self, point: Point) -> bool:
        return 1 <= point.row <= BOARD_SIZE and 1 <= point.col <= BOARD_SIZE

    @property
    def occupied_count(self) -> int:
        return len(self._grid)


class GomokuGameState:
    """Full game state for Gomoku (15x15, 5-in-a-row)."""

    def __init__(self) -> None:
        self.board = Board()
        self.current_player = Player.BLACK
        self.moves: list[Move] = []
        self._winner: Optional[Player] = None
        self._is_over = False

    @property
    def is_over(self) -> bool:
        return self._is_over

    @property
    def winner(self) -> Optional[Player]:
        return self._winner

    @property
    def is_draw(self) -> bool:
        return self._is_over and self._winner is None

    def legal_moves(self) -> list[Point]:
        if self._is_over:
            return []
        return [
            Point(r, c)
            for r in range(1, BOARD_SIZE + 1)
            for c in range(1, BOARD_SIZE + 1)
            if self.board.is_empty(Point(r, c))
        ]

    def apply_move(self, point: Point) -> None:
        """Place a stone for the current player and advance the turn."""
        assert not self._is_over, "Game is already over"
        assert self.board.is_on_grid(point), f"Point {point} is off the grid"
        assert self.board.is_empty(point), f"Point {format_point(point)} is occupied"

        player = self.current_player
        self.board.place(point, player)
        move = Move(point=point, player=player)
        self.moves.append(move)

        if self._check_win(point, player):
            self._winner = player
            self._is_over = True
        elif self.board.occupied_count == BOARD_SIZE * BOARD_SIZE:
            self._is_over = True

        self.current_player = self.current_player.other

    def undo_move(self) -> Optional[Move]:
        """Undo the last move. Returns the undone Move, or None if no moves."""
        if not self.moves:
            return None
        move = self.moves.pop()
        self.board.remove(move.point)
        self.current_player = move.player
        self._winner = None
        self._is_over = False
        return move

    def _check_win(self, point: Point, player: Player) -> bool:
        """Check if placing at `point` creates 5-in-a-row for `player`."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # Count forward
            for step in range(1, WIN_LENGTH):
                r, c = point.row + dr * step, point.col + dc * step
                p = Point(r, c)
                if not self.board.is_on_grid(p) or self.board.get(p) is not player:
                    break
                count += 1
            # Count backward
            for step in range(1, WIN_LENGTH):
                r, c = point.row - dr * step, point.col - dc * step
                p = Point(r, c)
                if not self.board.is_on_grid(p) or self.board.get(p) is not player:
                    break
                count += 1
            if count >= WIN_LENGTH:
                return True
        return False
