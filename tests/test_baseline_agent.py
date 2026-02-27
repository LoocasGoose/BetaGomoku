"""Tests for the baseline negamax agent."""

import pytest

from betagomoku.agent.baseline_agent import (
    BaselineAgent,
    evaluate,
    generate_candidates,
    order_moves,
    _pattern_score,
)
from betagomoku.game.board import BOARD_SIZE, GomokuGameState
from betagomoku.game.types import Player, Point


# ---------------------------------------------------------------------------
# Pattern scoring
# ---------------------------------------------------------------------------

class TestPatternScoring:
    def test_five_in_a_row(self):
        assert _pattern_score(5, 0) == 100_000
        assert _pattern_score(5, 2) == 100_000
        assert _pattern_score(6, 1) == 100_000

    def test_open_four(self):
        assert _pattern_score(4, 2) == 50_000

    def test_dead_pattern(self):
        assert _pattern_score(4, 0) == 0
        assert _pattern_score(3, 0) == 0
        assert _pattern_score(1, 0) == 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_empty_board_is_zero(self):
        gs = GomokuGameState()
        assert evaluate(gs) == 0

    def test_black_win_is_positive(self):
        gs = GomokuGameState()
        # Black wins with 5 in a row horizontally on row 1
        for c in range(1, 6):
            gs.apply_move(Point(1, c))  # black
            if c < 5:
                gs.apply_move(Point(2, c))  # white
        assert gs.is_over
        assert gs.winner is Player.BLACK
        assert evaluate(gs) == 1_000_000

    def test_white_win_is_negative(self):
        gs = GomokuGameState()
        # Set up so white wins
        gs.apply_move(Point(1, 1))   # B
        gs.apply_move(Point(2, 1))   # W
        gs.apply_move(Point(1, 2))   # B
        gs.apply_move(Point(2, 2))   # W
        gs.apply_move(Point(1, 3))   # B
        gs.apply_move(Point(2, 3))   # W
        gs.apply_move(Point(1, 4))   # B
        gs.apply_move(Point(2, 4))   # W
        gs.apply_move(Point(3, 1))   # B (not in a row)
        gs.apply_move(Point(2, 5))   # W wins
        assert gs.is_over
        assert gs.winner is Player.WHITE
        assert evaluate(gs) == -1_000_000

    def test_black_advantage_positive(self):
        gs = GomokuGameState()
        # Black has 3 in a row open, white has nothing special
        gs.apply_move(Point(8, 8))   # B
        gs.apply_move(Point(1, 1))   # W
        gs.apply_move(Point(8, 9))   # B
        gs.apply_move(Point(1, 2))   # W
        gs.apply_move(Point(8, 10))  # B
        # Black has open three at (8,8)-(8,10), should be positive
        assert evaluate(gs) > 0


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

class TestCandidateGeneration:
    def test_empty_board_returns_center(self):
        gs = GomokuGameState()
        candidates = generate_candidates(gs)
        center = (BOARD_SIZE + 1) // 2
        assert candidates == [Point(center, center)]

    def test_candidates_near_stones(self):
        gs = GomokuGameState()
        gs.apply_move(Point(8, 8))
        candidates = generate_candidates(gs)
        # All candidates should be within Chebyshev distance 2 of (8,8)
        for pt in candidates:
            assert abs(pt.row - 8) <= 2
            assert abs(pt.col - 8) <= 2

    def test_no_occupied_in_candidates(self):
        gs = GomokuGameState()
        gs.apply_move(Point(8, 8))
        gs.apply_move(Point(8, 9))
        candidates = generate_candidates(gs)
        occupied = {Point(8, 8), Point(8, 9)}
        for pt in candidates:
            assert pt not in occupied

    def test_candidates_on_grid(self):
        gs = GomokuGameState()
        gs.apply_move(Point(1, 1))  # corner
        candidates = generate_candidates(gs)
        for pt in candidates:
            assert 1 <= pt.row <= BOARD_SIZE
            assert 1 <= pt.col <= BOARD_SIZE


# ---------------------------------------------------------------------------
# Search behavior
# ---------------------------------------------------------------------------

class TestSearch:
    def test_finds_winning_move(self):
        """Agent should complete 5-in-a-row when possible."""
        gs = GomokuGameState()
        # Black has 4 in a row: (8,7), (8,8), (8,9), (8,10)
        # White stones elsewhere
        gs.apply_move(Point(8, 7))   # B
        gs.apply_move(Point(1, 1))   # W
        gs.apply_move(Point(8, 8))   # B
        gs.apply_move(Point(1, 2))   # W
        gs.apply_move(Point(8, 9))   # B
        gs.apply_move(Point(1, 3))   # W
        gs.apply_move(Point(8, 10))  # B
        gs.apply_move(Point(1, 4))   # W
        # Black to play — should play (8,11) or (8,6) to win
        agent = BaselineAgent(depth=2)
        move = agent.select_move(gs)
        assert move in (Point(8, 11), Point(8, 6))

    def test_blocks_opponent_win(self):
        """Agent should block opponent's 4-in-a-row."""
        gs = GomokuGameState()
        # Black plays at (8,7), (8,8), (8,9), (8,10) — open four
        # White needs to block
        gs.apply_move(Point(8, 7))   # B
        gs.apply_move(Point(1, 1))   # W
        gs.apply_move(Point(8, 8))   # B
        gs.apply_move(Point(1, 2))   # W
        gs.apply_move(Point(8, 9))   # B
        gs.apply_move(Point(1, 3))   # W
        gs.apply_move(Point(8, 10))  # B
        # White to play — must block at (8,6) or (8,11)
        agent = BaselineAgent(depth=2)
        move = agent.select_move(gs)
        assert move in (Point(8, 6), Point(8, 11))

    def test_returns_legal_move(self):
        gs = GomokuGameState()
        agent = BaselineAgent(depth=2)
        move = agent.select_move(gs)
        assert gs.board.is_on_grid(move)
        assert gs.board.is_empty(move)


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

class TestAgentConfig:
    def test_default_depth(self):
        agent = BaselineAgent()
        assert agent.depth == 2

    def test_custom_depth(self):
        agent = BaselineAgent(depth=3)
        assert agent.depth == 3

    def test_name(self):
        agent = BaselineAgent(depth=2)
        assert agent.name == "BaselineAgent(d=2)"

    def test_name_custom_depth(self):
        agent = BaselineAgent(depth=4)
        assert agent.name == "BaselineAgent(d=4)"
