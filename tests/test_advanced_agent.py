"""Tests for AdvancedAgent and its supporting functions."""

from __future__ import annotations

import pytest

from betagomoku.agent.advanced_agent import (
    AdvancedAgent,
    _compute_hash,
    _four_extension_squares,
    _is_winning_placement,
    _move_heuristic,
    _pattern_score,
    evaluate,
    generate_candidates,
    order_moves,
)
from betagomoku.agent.random_agent import RandomAgent
from betagomoku.game.board import BOARD_SIZE, GomokuGameState, parse_coordinate
from betagomoku.game.types import Player, Point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(*coords: str, first_player: Player = Player.BLACK) -> GomokuGameState:
    """Build a GomokuGameState by placing stones at given coordinates alternately."""
    gs = GomokuGameState()
    # Temporarily override current_player for construction
    for coord in coords:
        pt = parse_coordinate(coord)
        assert pt is not None, f"Bad coord: {coord}"
        gs.apply_move(pt)
    return gs


# ---------------------------------------------------------------------------
# Pattern score
# ---------------------------------------------------------------------------

class TestPatternScore:
    def test_five_is_always_max(self):
        assert _pattern_score(5, 0) == 100_000
        assert _pattern_score(5, 1) == 100_000
        assert _pattern_score(5, 2) == 100_000
        assert _pattern_score(6, 2) == 100_000  # overshoot

    def test_open_four(self):
        assert _pattern_score(4, 2) == 50_000

    def test_half_open_four(self):
        assert _pattern_score(4, 1) == 12_000

    def test_dead_four(self):
        assert _pattern_score(4, 0) == 0

    def test_open_three(self):
        assert _pattern_score(3, 2) == 6_000

    def test_unknown_patterns_are_zero(self):
        assert _pattern_score(1, 2) == 0
        assert _pattern_score(1, 1) == 0


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_empty_board_is_zero(self):
        gs = GomokuGameState()
        assert evaluate(gs) == 0

    def test_black_win(self):
        # Black wins: five in a row
        gs = make_state("E5", "A1", "F5", "A2", "G5", "A3", "H5", "A4", "I5")
        assert evaluate(gs) == 1_000_000

    def test_white_win(self):
        gs = make_state("A1", "E5", "A2", "F5", "A3", "G5", "A4", "H5", "B1", "I5")
        assert evaluate(gs) == -1_000_000

    def test_black_advantage_is_positive(self):
        # Black has a strong connected group, white has nothing
        gs = make_state("E5", "A1", "F5", "A2", "G5", "A3", "H5", "A4")
        # Black has open-4 at E-H5; white has scattered pieces at A1-4
        score = evaluate(gs)
        assert score > 0

    def test_fork_bonus_double_open3(self):
        """Two simultaneous open-3s for black should add FORK_BONUS."""
        # Black: horizontal open-3 at row 5, cols 5-7
        #        vertical  open-3 at col 5, rows 5-7
        # White: scattered non-threatening stones in opposite corner
        gs = GomokuGameState()
        moves_b = [Point(5, 5), Point(5, 6), Point(5, 7),  # horizontal open-3
                   Point(6, 5), Point(7, 5)]                  # vertical open-3
        # White stones spread out so they form no consecutive groups
        moves_w = [Point(15, 1), Point(15, 3), Point(15, 5), Point(15, 7), Point(15, 9)]
        for b, w in zip(moves_b, moves_w):
            gs.apply_move(b)
            gs.apply_move(w)
        score_with_fork = evaluate(gs)
        # Score should be positive (black advantage) and include fork bonus
        assert score_with_fork > 0


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

class TestGenerateCandidates:
    def test_empty_board_returns_center(self):
        gs = GomokuGameState()
        cands = generate_candidates(gs)
        center = (BOARD_SIZE + 1) // 2
        assert cands == [Point(center, center)]

    def test_all_candidates_are_empty(self):
        gs = make_state("E5", "A1")
        cands = generate_candidates(gs)
        for pt in cands:
            assert gs.board.is_empty(pt), f"{pt} is not empty"

    def test_all_candidates_on_grid(self):
        gs = make_state("A1", "O15")
        cands = generate_candidates(gs)
        for pt in cands:
            assert gs.board.is_on_grid(pt)

    def test_candidates_near_stones(self):
        gs = make_state("H8")  # one black stone at center
        cands = generate_candidates(gs)
        # All candidates should be within Chebyshev distance 2 of H8
        for pt in cands:
            assert abs(pt.row - 8) <= 2 and abs(pt.col - 8) <= 2

    def test_forced_response_when_opponent_has_four(self):
        """If opponent (white) has a 4-in-a-row, candidates should be forced responses."""
        gs = GomokuGameState()
        # Black plays somewhere irrelevant, then white builds a horizontal four
        gs.apply_move(Point(1, 1))   # black
        gs.apply_move(Point(5, 5))   # white
        gs.apply_move(Point(1, 2))   # black
        gs.apply_move(Point(5, 6))   # white
        gs.apply_move(Point(1, 3))   # black
        gs.apply_move(Point(5, 7))   # white
        gs.apply_move(Point(1, 4))   # black
        gs.apply_move(Point(5, 8))   # white — white now has open-4 at row 5
        # Black's candidates should include the blocking squares (5,4) and (5,9)
        cands = generate_candidates(gs)
        blocking = {Point(5, 4), Point(5, 9)}
        assert any(pt in blocking for pt in cands), \
            f"Expected blocking squares in candidates, got {cands}"

    def test_forced_response_when_opponent_has_broken_four(self):
        """If opponent has a broken-four winning move, force blocking squares."""
        gs = GomokuGameState()
        # White stones at (8,7), (8,8), (8,10), (8,11) -> winning gap at (8,9)
        gs.apply_move(Point(3, 3))   # black
        gs.apply_move(Point(8, 7))   # white
        gs.apply_move(Point(3, 4))   # black
        gs.apply_move(Point(8, 8))   # white
        gs.apply_move(Point(4, 3))   # black
        gs.apply_move(Point(8, 10))  # white
        gs.apply_move(Point(4, 4))   # black
        gs.apply_move(Point(8, 11))  # white
        cands = generate_candidates(gs)
        assert Point(8, 9) in cands


# ---------------------------------------------------------------------------
# Four extension squares
# ---------------------------------------------------------------------------

class TestFourExtensionSquares:
    def test_horizontal_four(self):
        gs = GomokuGameState()
        # Black four at row 5, cols 5-8
        gs.apply_move(Point(5, 5))
        gs.apply_move(Point(1, 1))
        gs.apply_move(Point(5, 6))
        gs.apply_move(Point(1, 2))
        gs.apply_move(Point(5, 7))
        gs.apply_move(Point(1, 3))
        gs.apply_move(Point(5, 8))
        gs.apply_move(Point(1, 4))
        # White's turn — black has a four
        squares = _four_extension_squares(gs, Player.BLACK)
        assert Point(5, 4) in squares or Point(5, 9) in squares

    def test_no_four(self):
        gs = make_state("E5")
        squares = _four_extension_squares(gs, Player.BLACK)
        assert squares == []


class TestImmediateWinningPlacement:
    def test_detects_gap_four_completion(self):
        gs = GomokuGameState()
        gs.apply_move(Point(3, 3))
        gs.apply_move(Point(8, 7))
        gs.apply_move(Point(3, 4))
        gs.apply_move(Point(8, 8))
        gs.apply_move(Point(4, 3))
        gs.apply_move(Point(8, 10))
        gs.apply_move(Point(4, 4))
        gs.apply_move(Point(8, 11))
        assert _is_winning_placement(gs, Point(8, 9), Player.WHITE)


# ---------------------------------------------------------------------------
# Zobrist hashing
# ---------------------------------------------------------------------------

class TestZobristHash:
    def test_empty_board_hash_is_deterministic(self):
        gs = GomokuGameState()
        assert _compute_hash(gs) == _compute_hash(gs)

    def test_different_positions_have_different_hashes(self):
        gs1 = make_state("E5")
        gs2 = make_state("F6")
        assert _compute_hash(gs1) != _compute_hash(gs2)

    def test_same_position_same_hash(self):
        gs1 = make_state("E5", "A1")
        gs2 = make_state("E5", "A1")
        assert _compute_hash(gs1) == _compute_hash(gs2)

    def test_side_to_move_affects_hash(self):
        # After one move, current player has changed
        gs = GomokuGameState()
        h_before = _compute_hash(gs)
        gs.apply_move(Point(8, 8))
        gs.apply_move(Point(1, 1))  # white plays, now it's black's turn
        # After 2 moves it's BLACK's turn again, but board is different
        h_after = _compute_hash(gs)
        assert h_before != h_after


# ---------------------------------------------------------------------------
# AdvancedAgent
# ---------------------------------------------------------------------------

class TestAdvancedAgent:
    def test_name(self):
        agent = AdvancedAgent(depth=6)
        assert "Advanced" in agent.name
        assert "6" in agent.name

    def test_default_depth_is_6(self):
        agent = AdvancedAgent()
        assert agent.depth == 6

    def test_returns_valid_move_empty_board(self):
        gs = GomokuGameState()
        agent = AdvancedAgent(depth=2)
        move = agent.select_move(gs)
        assert gs.board.is_on_grid(move)
        assert gs.board.is_empty(move)

    def test_returns_valid_move_mid_game(self):
        gs = make_state("E5", "F6", "G7", "H8")
        agent = AdvancedAgent(depth=2)
        move = agent.select_move(gs)
        assert gs.board.is_on_grid(move)
        assert gs.board.is_empty(move)

    def test_finds_win_in_one(self):
        """Agent must take winning move when available."""
        gs = GomokuGameState()
        # Black: E5 F5 G5 H5 — needs I5 to win
        gs.apply_move(Point(5, 5))  # black
        gs.apply_move(Point(1, 1))  # white
        gs.apply_move(Point(5, 6))  # black
        gs.apply_move(Point(1, 2))  # white
        gs.apply_move(Point(5, 7))  # black
        gs.apply_move(Point(1, 3))  # white
        gs.apply_move(Point(5, 8))  # black
        gs.apply_move(Point(1, 4))  # white
        # Black's turn — should play I5 = Point(5,9) to win
        agent = AdvancedAgent(depth=2)
        move = agent.select_move(gs)
        assert move == Point(5, 9), f"Expected Point(5,9) win, got {move}"

    def test_blocks_opponent_four(self):
        """Agent must block opponent's 4-in-a-row."""
        gs = GomokuGameState()
        # White: row 5, cols 5-8 (open-4)
        # Black stones are scattered so black cannot win in 1
        gs.apply_move(Point(3, 3))  # black
        gs.apply_move(Point(5, 5))  # white
        gs.apply_move(Point(3, 6))  # black (not adjacent to 3,3 diagonally by 3)
        gs.apply_move(Point(5, 6))  # white
        gs.apply_move(Point(9, 9))  # black (far away — no 4-in-a-row for black)
        gs.apply_move(Point(5, 7))  # white
        gs.apply_move(Point(9, 2))  # black
        gs.apply_move(Point(5, 8))  # white — white has open-4
        # Black must block at Point(5,4) or Point(5,9)
        agent = AdvancedAgent(depth=2)
        move = agent.select_move(gs)
        assert move in {Point(5, 4), Point(5, 9)}, \
            f"Expected blocking move, got {move}"

    def test_beats_random_agent(self):
        """AdvancedAgent should beat RandomAgent consistently."""
        advanced = AdvancedAgent(depth=4)
        random_agent = RandomAgent()
        wins = 0
        games = 5
        for _ in range(games):
            gs = GomokuGameState()
            while not gs.is_over:
                if gs.current_player is Player.BLACK:
                    move = advanced.select_move(gs)
                else:
                    move = random_agent.select_move(gs)
                gs.apply_move(move)
            if gs.winner is Player.BLACK:
                wins += 1
        assert wins >= 4, f"AdvancedAgent only won {wins}/{games} games vs RandomAgent"

    def test_first_move_is_center(self):
        """On an empty board, agent should play near center."""
        gs = GomokuGameState()
        agent = AdvancedAgent(depth=2)
        move = agent.select_move(gs)
        center = (BOARD_SIZE + 1) // 2
        # Should be at or very near center
        assert abs(move.row - center) <= 2 and abs(move.col - center) <= 2


# ---------------------------------------------------------------------------
# Order moves
# ---------------------------------------------------------------------------

class TestOrderMoves:
    def test_tt_move_comes_first(self):
        gs = make_state("E5", "A1")
        cands = generate_candidates(gs)
        tt_move = cands[-1]  # pick an arbitrary candidate as TT move
        ordered = order_moves(gs, cands, tt_move, [None, None], {})
        assert ordered[0] == tt_move

    def test_no_tt_move_still_orders(self):
        gs = make_state("E5", "A1")
        cands = generate_candidates(gs)
        ordered = order_moves(gs, cands, None, [None, None], {})
        assert len(ordered) == len(cands)
