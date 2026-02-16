"""Baseline agent: Negamax with alpha-beta pruning and pattern-based evaluation."""

from __future__ import annotations

import math
from typing import Optional

from betagomoku.agent.base import Agent
from betagomoku.game.board import BOARD_SIZE, WIN_LENGTH, GomokuGameState
from betagomoku.game.types import Player, Point

# ---------------------------------------------------------------------------
# Pattern scoring table: (consecutive_count, open_ends) -> score
# ---------------------------------------------------------------------------

PATTERN_SCORES: dict[tuple[int, int], int] = {
    (5, 0): 100_000,
    (5, 1): 100_000,
    (5, 2): 100_000,
    (4, 2): 10_000,
    (4, 1): 1_000,
    (3, 2): 1_000,
    (3, 1): 100,
    (2, 2): 100,
    (2, 1): 10,
}

INF = math.inf

# Four direction axes for scanning patterns
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _pattern_score(count: int, open_ends: int) -> int:
    """Look up score for a consecutive group with given open ends."""
    if count >= WIN_LENGTH:
        return 100_000
    return PATTERN_SCORES.get((count, open_ends), 0)


# ---------------------------------------------------------------------------
# Evaluation (absolute viewpoint: positive = BLACK advantage)
# ---------------------------------------------------------------------------

def evaluate(game_state: GomokuGameState) -> int:
    """Static evaluation of the board position.

    Returns a positive score if BLACK is ahead, negative if WHITE is ahead.
    Terminal states return +/-1,000,000 for wins, 0 for draws.
    """
    if game_state.is_over:
        if game_state.winner is Player.BLACK:
            return 1_000_000
        elif game_state.winner is Player.WHITE:
            return -1_000_000
        return 0  # draw

    board = game_state.board
    score = 0

    for dr, dc in DIRECTIONS:
        # Scan all cells in this direction axis.
        # Use a visited set so we only count each consecutive group once.
        visited: set[Point] = set()

        for r in range(1, BOARD_SIZE + 1):
            for c in range(1, BOARD_SIZE + 1):
                pt = Point(r, c)
                if pt in visited:
                    continue
                player = board.get(pt)
                if player is None:
                    continue

                # Count consecutive stones in this direction
                count = 0
                curr_r, curr_c = r, c
                while True:
                    p = Point(curr_r, curr_c)
                    if not board.is_on_grid(p) or board.get(p) is not player:
                        break
                    visited.add(p)
                    count += 1
                    curr_r += dr
                    curr_c += dc

                # Count open ends
                open_ends = 0
                # Check before the group
                before = Point(r - dr, c - dc)
                if board.is_on_grid(before) and board.get(before) is None:
                    open_ends += 1
                # Check after the group
                after = Point(r + dr * count, c + dc * count)
                if board.is_on_grid(after) and board.get(after) is None:
                    open_ends += 1

                pat_score = _pattern_score(count, open_ends)
                if player is Player.BLACK:
                    score += pat_score
                else:
                    score -= pat_score

    return score


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(game_state: GomokuGameState) -> list[Point]:
    """Return candidate moves near existing stones (Chebyshev distance <= 2).

    On an empty board, returns the center point.
    """
    if not game_state.moves:
        center = (BOARD_SIZE + 1) // 2
        return [Point(center, center)]

    board = game_state.board
    occupied = {m.point for m in game_state.moves}
    candidates: set[Point] = set()

    for pt in occupied:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = pt.row + dr, pt.col + dc
                np = Point(nr, nc)
                if board.is_on_grid(np) and board.is_empty(np):
                    candidates.add(np)

    return list(candidates)


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

def _move_heuristic(game_state: GomokuGameState, move: Point) -> int:
    """Fast heuristic for a candidate move: offensive pattern + 0.5 * defensive."""
    board = game_state.board
    current = game_state.current_player
    opponent = current.other
    score = 0

    for dr, dc in DIRECTIONS:
        for player, weight in ((current, 1), (opponent, 0.5)):
            # Count how many consecutive stones of `player` the move would extend
            count = 1  # the move itself
            open_ends = 0

            # Forward
            cr, cc = move.row + dr, move.col + dc
            while True:
                p = Point(cr, cc)
                if not board.is_on_grid(p) or board.get(p) is not player:
                    break
                count += 1
                cr += dr
                cc += dc
            # Check if forward end is open
            end_fwd = Point(cr, cc)
            if board.is_on_grid(end_fwd) and board.get(end_fwd) is None:
                open_ends += 1

            # Backward
            cr, cc = move.row - dr, move.col - dc
            while True:
                p = Point(cr, cc)
                if not board.is_on_grid(p) or board.get(p) is not player:
                    break
                count += 1
                cr -= dr
                cc -= dc
            # Check if backward end is open
            end_bwd = Point(cr, cc)
            if board.is_on_grid(end_bwd) and board.get(end_bwd) is None:
                open_ends += 1

            score += int(weight * _pattern_score(count, open_ends))

    return score


def order_moves(game_state: GomokuGameState, candidates: list[Point]) -> list[Point]:
    """Sort candidates by heuristic score (descending) for better pruning."""
    return sorted(candidates, key=lambda m: _move_heuristic(game_state, m), reverse=True)


# ---------------------------------------------------------------------------
# Negamax with alpha-beta
# ---------------------------------------------------------------------------

def negamax(game_state: GomokuGameState, depth: int, alpha: float, beta: float, color: int) -> float:
    """Negamax search with alpha-beta pruning.

    color is +1 for BLACK's turn, -1 for WHITE's turn.
    Returns the best score from the perspective of the current player.
    """
    if game_state.is_over or depth == 0:
        return color * evaluate(game_state)

    candidates = order_moves(game_state, generate_candidates(game_state))
    if not candidates:
        return color * evaluate(game_state)

    best = -INF
    for move in candidates:
        game_state.apply_move(move)
        score = -negamax(game_state, depth - 1, -beta, -alpha, -color)
        game_state.undo_move()

        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best


# ---------------------------------------------------------------------------
# BaselineAgent
# ---------------------------------------------------------------------------

class BaselineAgent(Agent):
    """Negamax + alpha-beta agent with pattern-based evaluation."""

    def __init__(self, depth: int = 2) -> None:
        self.depth = depth

    @property
    def name(self) -> str:
        return f"BaselineAgent(d={self.depth})"

    def select_move(self, game_state: GomokuGameState) -> Point:
        color = 1 if game_state.current_player is Player.BLACK else -1
        candidates = order_moves(game_state, generate_candidates(game_state))

        # Check for immediate winning move
        for move in candidates:
            game_state.apply_move(move)
            if game_state.is_over and game_state.winner is game_state.current_player.other:
                game_state.undo_move()
                return move
            game_state.undo_move()

        best_score = -INF
        best_move: Optional[Point] = None

        for move in candidates:
            game_state.apply_move(move)
            score = -negamax(game_state, self.depth - 1, -INF, INF, -color)
            game_state.undo_move()

            if score > best_score:
                best_score = score
                best_move = move

        assert best_move is not None, "No candidates found"
        return best_move
