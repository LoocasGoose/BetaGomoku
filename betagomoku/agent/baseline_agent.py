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
    (4, 2): 50_000,   # open four — practically a win
    (4, 1): 8_000,    # half-open four — forcing, opponent must respond
    (3, 2): 15_000,    # open three — creates open four next move
    (3, 1): 1_000,    # half-open three
    (2, 2): 1_000,    # open two
    (2, 1): 100,      # half-open two
}

INF = math.inf

# Four direction axes for scanning patterns
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# Max candidates to evaluate at each search depth
MAX_CANDIDATES = 20


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

    # Only scan from occupied cells rather than the full 15x15 grid
    occupied = {m.point for m in game_state.moves}

    for dr, dc in DIRECTIONS:
        visited: set[Point] = set()

        for pt in occupied:
            if pt in visited:
                continue
            player = board.get(pt)
            if player is None:
                continue

            # Walk backward to find the start of the group in this direction
            sr, sc = pt.row, pt.col
            while True:
                pr, pc = sr - dr, sc - dc
                p = Point(pr, pc)
                if not board.is_on_grid(p) or board.get(p) is not player:
                    break
                sr, sc = pr, pc

            start = Point(sr, sc)
            if start in visited:
                continue

            # Count consecutive stones in this direction from start
            count = 0
            curr_r, curr_c = sr, sc
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
            before = Point(sr - dr, sc - dc)
            if board.is_on_grid(before) and board.get(before) is None:
                open_ends += 1
            after = Point(sr + dr * count, sc + dc * count)
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
# Negamax with alpha-beta + transposition table
# ---------------------------------------------------------------------------

def negamax(
    game_state: GomokuGameState,
    depth: int,
    alpha: float,
    beta: float,
    color: int,
    tt: Optional[dict] = None,
) -> float:
    """Negamax search with alpha-beta pruning and transposition table.

    color is +1 for BLACK's turn, -1 for WHITE's turn.
    Returns the best score from the perspective of the current player.
    """
    if game_state.is_over or depth == 0:
        return color * evaluate(game_state)

    # Transposition table lookup
    if tt is not None:
        key = _tt_key(game_state)
        entry = tt.get(key)
        if entry is not None:
            tt_depth, tt_flag, tt_score = entry
            if tt_depth >= depth:
                if tt_flag == 0:  # exact
                    return tt_score
                elif tt_flag == 1 and tt_score >= beta:  # lower bound
                    return tt_score
                elif tt_flag == -1 and tt_score <= alpha:  # upper bound
                    return tt_score

    candidates = order_moves(game_state, generate_candidates(game_state))
    if not candidates:
        return color * evaluate(game_state)

    # Cap candidates to limit branching factor
    if len(candidates) > MAX_CANDIDATES:
        candidates = candidates[:MAX_CANDIDATES]

    orig_alpha = alpha
    best = -INF
    for move in candidates:
        game_state.apply_move(move)
        score = -negamax(game_state, depth - 1, -beta, -alpha, -color, tt)
        game_state.undo_move()

        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    # Store in transposition table
    if tt is not None:
        if best <= orig_alpha:
            flag = -1  # upper bound
        elif best >= beta:
            flag = 1   # lower bound
        else:
            flag = 0   # exact
        tt[key] = (depth, flag, best)

    return best


def _tt_key(game_state: GomokuGameState) -> tuple:
    """Create a hashable key for the transposition table from the board state."""
    return (
        frozenset(game_state.board._grid.items()),
        game_state.current_player,
    )


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

        # Cap root candidates too
        if len(candidates) > MAX_CANDIDATES:
            candidates = candidates[:MAX_CANDIDATES]

        tt: dict = {}
        best_score = -INF
        best_move: Optional[Point] = None

        for move in candidates:
            game_state.apply_move(move)
            score = -negamax(game_state, self.depth - 1, -INF, -best_score, -color, tt)
            game_state.undo_move()

            if score > best_score:
                best_score = score
                best_move = move

        assert best_move is not None, "No candidates found"
        return best_move
