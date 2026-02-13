import pytest

from betagomoku.game.board import (
    BOARD_SIZE,
    Board,
    GomokuGameState,
    format_point,
    parse_coordinate,
)
from betagomoku.game.types import Player, Point


class TestParseCoordinate:
    def test_valid(self):
        assert parse_coordinate("A1") == Point(1, 1)
        assert parse_coordinate("E5") == Point(5, 5)
        assert parse_coordinate("I9") == Point(9, 9)
        assert parse_coordinate("e5") == Point(5, 5)  # case insensitive

    def test_invalid(self):
        assert parse_coordinate("") is None
        assert parse_coordinate("Z1") is None
        assert parse_coordinate("A0") is None
        assert parse_coordinate("A10") is None
        assert parse_coordinate("XX") is None


class TestFormatPoint:
    def test_basic(self):
        assert format_point(Point(1, 1)) == "A1"
        assert format_point(Point(5, 5)) == "E5"
        assert format_point(Point(9, 9)) == "I9"


class TestBoard:
    def test_place_and_get(self):
        b = Board()
        p = Point(3, 4)
        b.place(p, Player.BLACK)
        assert b.get(p) is Player.BLACK
        assert not b.is_empty(p)

    def test_remove(self):
        b = Board()
        p = Point(3, 4)
        b.place(p, Player.BLACK)
        b.remove(p)
        assert b.is_empty(p)

    def test_is_on_grid(self):
        b = Board()
        assert b.is_on_grid(Point(1, 1))
        assert b.is_on_grid(Point(9, 9))
        assert not b.is_on_grid(Point(0, 1))
        assert not b.is_on_grid(Point(1, 10))


class TestGomokuGameState:
    def test_initial_state(self):
        g = GomokuGameState()
        assert g.current_player is Player.BLACK
        assert not g.is_over
        assert g.winner is None
        assert len(g.legal_moves()) == BOARD_SIZE * BOARD_SIZE

    def test_alternating_turns(self):
        g = GomokuGameState()
        g.apply_move(Point(5, 5))
        assert g.current_player is Player.WHITE
        g.apply_move(Point(5, 6))
        assert g.current_player is Player.BLACK

    def test_horizontal_win(self):
        g = GomokuGameState()
        # Black: row 1, cols 1-5. White: row 2, cols 1-4.
        for i in range(4):
            g.apply_move(Point(1, i + 1))  # Black
            g.apply_move(Point(2, i + 1))  # White
        g.apply_move(Point(1, 5))  # Black wins
        assert g.is_over
        assert g.winner is Player.BLACK

    def test_vertical_win(self):
        g = GomokuGameState()
        for i in range(4):
            g.apply_move(Point(i + 1, 1))  # Black
            g.apply_move(Point(i + 1, 2))  # White
        g.apply_move(Point(5, 1))  # Black wins
        assert g.is_over
        assert g.winner is Player.BLACK

    def test_diagonal_win(self):
        g = GomokuGameState()
        # Black on main diagonal (1,1)-(5,5), White on col 9
        moves_black = [Point(i, i) for i in range(1, 6)]
        moves_white = [Point(i, 9) for i in range(1, 5)]
        for i in range(4):
            g.apply_move(moves_black[i])
            g.apply_move(moves_white[i])
        g.apply_move(moves_black[4])  # Black wins
        assert g.is_over
        assert g.winner is Player.BLACK

    def test_anti_diagonal_win(self):
        g = GomokuGameState()
        # Black: (1,5),(2,4),(3,3),(4,2),(5,1). White: col 9.
        moves_black = [Point(i, 6 - i) for i in range(1, 6)]
        moves_white = [Point(i, 9) for i in range(1, 5)]
        for i in range(4):
            g.apply_move(moves_black[i])
            g.apply_move(moves_white[i])
        g.apply_move(moves_black[4])
        assert g.is_over
        assert g.winner is Player.BLACK

    def test_no_premature_win(self):
        """4 in a row should NOT trigger a win."""
        g = GomokuGameState()
        for i in range(4):
            g.apply_move(Point(1, i + 1))  # Black
            g.apply_move(Point(2, i + 1))  # White
        # Black has 4 in a row on row 1, but not 5
        assert not g.is_over

    def test_undo_move(self):
        g = GomokuGameState()
        g.apply_move(Point(5, 5))
        g.apply_move(Point(5, 6))
        move = g.undo_move()
        assert move is not None
        assert move.point == Point(5, 6)
        assert g.current_player is Player.WHITE
        assert g.board.is_empty(Point(5, 6))

    def test_undo_reverses_win(self):
        g = GomokuGameState()
        for i in range(4):
            g.apply_move(Point(1, i + 1))
            g.apply_move(Point(2, i + 1))
        g.apply_move(Point(1, 5))  # Black wins
        assert g.is_over

        g.undo_move()
        assert not g.is_over
        assert g.winner is None

    def test_undo_empty_returns_none(self):
        g = GomokuGameState()
        assert g.undo_move() is None

    def test_cannot_play_on_occupied(self):
        g = GomokuGameState()
        g.apply_move(Point(5, 5))
        with pytest.raises(AssertionError):
            g.apply_move(Point(5, 5))

    def test_cannot_play_after_game_over(self):
        g = GomokuGameState()
        for i in range(4):
            g.apply_move(Point(1, i + 1))
            g.apply_move(Point(2, i + 1))
        g.apply_move(Point(1, 5))  # Black wins
        with pytest.raises(AssertionError):
            g.apply_move(Point(3, 1))

    def test_legal_moves_empty_after_game_over(self):
        g = GomokuGameState()
        for i in range(4):
            g.apply_move(Point(1, i + 1))
            g.apply_move(Point(2, i + 1))
        g.apply_move(Point(1, 5))
        assert g.legal_moves() == []
