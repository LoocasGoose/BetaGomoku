"""Advanced agent: Optimized alpha-beta with PVS, iterative deepening,
Zobrist transposition table, killer moves, history heuristic, fork detection,
quiescence search, LMR, broken-four evaluation, and open-3 forced response.

This is a separate copy of the baseline agent architecture, fixed at depth 6
and enhanced with every major alpha-beta optimization from Gomoku/chess programming
research. The baseline_agent.py is left entirely unchanged.

Key improvements over BaselineAgent:
  1. Zobrist hashing       — O(1) incremental TT keys instead of O(n) frozenset
  2. TT best-move storage  — reuse best move from shallower iterations for ordering
  3. Iterative deepening   — search depths 1..N; each depth improves ordering for next
  4. Aspiration windows    — narrow alpha-beta window around previous iteration's score
  5. PVS (NegaScout)       — null window on non-first moves; 10-20% fewer nodes
  6. Killer moves          — 2 killers per depth; moves that caused recent beta cutoffs
  7. History heuristic     — global move-effectiveness table; tiebreaker in ordering
  8. Tiered move ordering  — TT move > wins/blocks > killers > history+heuristic
  9. Forced-response       — restrict candidates when opponent has four OR double open-3
 10. Fork detection        — evaluation bonus for double open-3 or four+open-3
 11. Quiescence search     — continue forcing moves beyond depth=0 to avoid horizon effect
 12. LMR                   — reduce depth for late non-forcing moves, re-search if needed
 13. Broken-four eval      — score non-contiguous 4-in-5-window patterns in static eval
"""

from __future__ import annotations

import random
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
    (4, 2): 50_000,   # open four — unstoppable win
    (4, 1): 12_000,   # half-open four — must block
    (3, 2): 6_000,    # open three — creates open four
    (3, 1): 1_500,    # half-open three
    (2, 2): 1_000,    # open two
    (2, 1): 100,      # half-open two
}

# Integer infinity (all evaluations are bounded by WIN_SCORE)
INF = 10_000_000

# Four direction axes for scanning patterns
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# Candidate move caps
MAX_CANDIDATES_ROOT = 30
MAX_CANDIDATES_INNER = 20

# Aspiration window: search depth ≥ 3 uses prev_score ± WINDOW first
ASPIRATION_WINDOW = 500

# Fork bonuses added to static evaluation
FORK_BONUS = 5_000            # two simultaneous open-3s (double threat)
DOUBLE_THREAT_BONUS = 9_000   # four + open-3 (even stronger fork)

# Preserve forcing moves even when candidate lists are capped
FORCING_MOVE_HEURISTIC = 12_000

# ---------------------------------------------------------------------------
# New accuracy-improvement constants
# ---------------------------------------------------------------------------

# Quiescence search: extend depth=0 leaves by this many extra plies,
# searching only forcing moves to avoid the horizon effect.
QUIESCENCE_DEPTH = 2
MAX_QUIESCENCE_FORCING = 5    # cap on forcing moves explored per quiescence node

# Late Move Reductions (LMR): reduce depth-1 for non-forcing moves at late indices.
LMR_MIN_DEPTH = 3             # minimum remaining depth to apply LMR
LMR_MIN_INDEX = 4             # minimum move index (0-based) to apply LMR
LMR_QUIET_THRESHOLD = 6_000   # moves with heuristic >= this are never LMR-reduced

# Broken-four evaluation: bonus per 4-in-5-window pattern (non-contiguous).
# These are direct win-in-1 threats (the gap square wins immediately) that the
# contiguous group evaluator misses.  Scored similar to a half-open four.
BROKEN_FOUR_SCORE = 10_000

# ---------------------------------------------------------------------------
# Zobrist hashing (module-level, deterministic seed)
# ---------------------------------------------------------------------------

_rng = random.Random(42)

# ZOBRIST[row][col][player.value]: player.value = 1 (BLACK) or 2 (WHITE)
ZOBRIST: list[list[list[int]]] = [
    [[_rng.getrandbits(64) for _ in range(3)] for _ in range(BOARD_SIZE + 2)]
    for _ in range(BOARD_SIZE + 2)
]
ZOBRIST_SIDE: int = _rng.getrandbits(64)  # XOR'd when it is WHITE's turn


def _compute_hash(game_state: GomokuGameState) -> int:
    """Compute the Zobrist hash from scratch for the current board + side to move."""
    h = 0
    for pt, player in game_state.board._grid.items():
        h ^= ZOBRIST[pt.row][pt.col][player.value]
    if game_state.current_player is Player.WHITE:
        h ^= ZOBRIST_SIDE
    return h


# ---------------------------------------------------------------------------
# Pattern utilities
# ---------------------------------------------------------------------------

def _pattern_score(count: int, open_ends: int) -> int:
    """Look up score for a consecutive group with given open ends."""
    if count >= WIN_LENGTH:
        return 100_000
    return PATTERN_SCORES.get((count, open_ends), 0)


# ---------------------------------------------------------------------------
# Broken-four bonus (non-contiguous 4-in-5-window patterns)
# ---------------------------------------------------------------------------

def _broken_four_bonus(board, occupied: set[Point]) -> int:
    """Score broken-four patterns: 4 player stones in a 5-cell window with 1 gap.

    A broken four like XX_XX is a direct win-in-1 threat (filling the gap wins
    immediately) that the contiguous-group evaluator does not score.  Only counts
    windows starting at an occupied stone to avoid double-counting.

    Returns positive for BLACK advantage, negative for WHITE advantage.
    """
    if len(occupied) < 4:
        return 0

    bonus = 0
    for dr, dc in DIRECTIONS:
        for pt in occupied:
            # Build a 5-cell window starting at this stone in direction (dr, dc)
            cells: list[Optional[Player]] = []
            r, c = pt.row, pt.col
            valid = True
            for _ in range(5):
                p = Point(r, c)
                if not board.is_on_grid(p):
                    valid = False
                    break
                cells.append(board.get(p))
                r += dr
                c += dc

            if not valid:
                continue

            for player in (Player.BLACK, Player.WHITE):
                # Only count windows where this stone belongs to 'player'
                if cells[0] is not player:
                    continue

                player_count = sum(1 for x in cells if x is player)
                opp_count = sum(1 for x in cells if x is not None and x is not player)

                if player_count != 4 or opp_count > 0:
                    continue

                # Skip contiguous-4 patterns already scored by the main evaluator.
                # Contiguous iff the 4 stone positions span exactly 4 cells.
                indices = [i for i, x in enumerate(cells) if x is player]
                if indices[-1] - indices[0] + 1 == 4:
                    continue  # e.g., XXXX_ — already captured as (4, open_ends)

                sign = 1 if player is Player.BLACK else -1
                bonus += sign * BROKEN_FOUR_SCORE

    return bonus


# ---------------------------------------------------------------------------
# Evaluation with fork detection
# ---------------------------------------------------------------------------

def evaluate(game_state: GomokuGameState) -> int:
    """Static evaluation. Positive = BLACK advantage.

    Terminal states: ±1_000_000 for wins, 0 for draw.
    Includes fork bonuses for double-threat positions and broken-four patterns.
    """
    if game_state.is_over:
        if game_state.winner is Player.BLACK:
            return 1_000_000
        elif game_state.winner is Player.WHITE:
            return -1_000_000
        return 0

    board = game_state.board
    occupied = {m.point for m in game_state.moves}
    score = 0

    # Counters for fork bonus detection
    black_open3 = 0
    black_four = 0
    white_open3 = 0
    white_four = 0

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

            if Point(sr, sc) in visited:
                continue

            # Count consecutive stones in the forward direction
            count = 0
            cr, cc = sr, sc
            while True:
                p = Point(cr, cc)
                if not board.is_on_grid(p) or board.get(p) is not player:
                    break
                visited.add(p)
                count += 1
                cr += dr
                cc += dc

            # Count open ends
            open_ends = 0
            before = Point(sr - dr, sc - dc)
            if board.is_on_grid(before) and board.get(before) is None:
                open_ends += 1
            after = Point(sr + dr * count, sc + dc * count)
            if board.is_on_grid(after) and board.get(after) is None:
                open_ends += 1

            ps = _pattern_score(count, open_ends)
            if player is Player.BLACK:
                score += ps
                if count == 3 and open_ends == 2:
                    black_open3 += 1
                elif count >= 4:
                    black_four += 1
            else:
                score -= ps
                if count == 3 and open_ends == 2:
                    white_open3 += 1
                elif count >= 4:
                    white_four += 1

    # Fork bonuses: reward (penalize) positions with multiple simultaneous threats
    if black_open3 >= 2:
        score += FORK_BONUS          # double open-3: two threats, only one can be blocked
    if white_open3 >= 2:
        score -= FORK_BONUS
    if black_four >= 1 and black_open3 >= 1:
        score += DOUBLE_THREAT_BONUS  # four + open-3: defender is overwhelmed
    if white_four >= 1 and white_open3 >= 1:
        score -= DOUBLE_THREAT_BONUS

    # Broken-four bonus: score non-contiguous 4-in-5-window winning threats
    score += _broken_four_bonus(board, occupied)

    return score


# ---------------------------------------------------------------------------
# Forced-response detection
# ---------------------------------------------------------------------------

def _four_extension_squares(game_state: GomokuGameState, player: Player) -> list[Point]:
    """Return all empty squares that would complete (or extend) player's 4-in-a-row.

    These squares are the critical blocking/winning positions when a four exists.
    """
    board = game_state.board
    occupied = {m.point for m in game_state.moves}
    squares: list[Point] = []

    for dr, dc in DIRECTIONS:
        visited: set[Point] = set()

        for pt in occupied:
            if pt in visited:
                continue
            if board.get(pt) is not player:
                continue

            # Walk backward to group start
            sr, sc = pt.row, pt.col
            while True:
                pr, pc = sr - dr, sc - dc
                p = Point(pr, pc)
                if not board.is_on_grid(p) or board.get(p) is not player:
                    break
                sr, sc = pr, pc

            if Point(sr, sc) in visited:
                continue

            # Count consecutive forward
            count = 0
            cr, cc = sr, sc
            while True:
                p = Point(cr, cc)
                if not board.is_on_grid(p) or board.get(p) is not player:
                    break
                visited.add(p)
                count += 1
                cr += dr
                cc += dc

            # Only care about exactly-4 groups (5+ is already a win)
            if count == 4:
                before = Point(sr - dr, sc - dc)
                if board.is_on_grid(before) and board.is_empty(before):
                    squares.append(before)
                after = Point(cr, cc)  # one past the last stone
                if board.is_on_grid(after) and board.is_empty(after):
                    squares.append(after)

    return squares


def _open_three_squares(
    game_state: GomokuGameState, player: Player
) -> tuple[list[Point], int]:
    """Return (blocking end-squares, count) for player's open-3 patterns.

    An open-3 is exactly 3 consecutive stones with both ends open.
    Returns the endpoint squares that would block one of these threats, and the
    total number of distinct open-3 patterns found across all directions.
    """
    board = game_state.board
    occupied = {m.point for m in game_state.moves}
    squares: list[Point] = []
    count = 0

    for dr, dc in DIRECTIONS:
        visited: set[Point] = set()

        for pt in occupied:
            if pt in visited or board.get(pt) is not player:
                continue

            # Walk backward to group start
            sr, sc = pt.row, pt.col
            while True:
                pr, pc = sr - dr, sc - dc
                p = Point(pr, pc)
                if not board.is_on_grid(p) or board.get(p) is not player:
                    break
                sr, sc = pr, pc

            if Point(sr, sc) in visited:
                continue

            # Count consecutive forward
            n = 0
            cr, cc = sr, sc
            while True:
                p = Point(cr, cc)
                if not board.is_on_grid(p) or board.get(p) is not player:
                    break
                visited.add(p)
                n += 1
                cr += dr
                cc += dc

            if n == 3:
                before = Point(sr - dr, sc - dc)
                after = Point(cr, cc)
                open_before = board.is_on_grid(before) and board.is_empty(before)
                open_after = board.is_on_grid(after) and board.is_empty(after)
                if open_before and open_after:
                    count += 1
                    squares.append(before)
                    squares.append(after)

    return squares, count


def _is_winning_placement(game_state: GomokuGameState, move: Point, player: Player) -> bool:
    """Return True if placing `player` at `move` would immediately make 5-in-a-row.

    This catches both contiguous fours and gapped threats (e.g. XX_XX).
    """
    board = game_state.board
    if not board.is_on_grid(move) or not board.is_empty(move):
        return False

    for dr, dc in DIRECTIONS:
        count = 1

        # Forward
        r, c = move.row + dr, move.col + dc
        while True:
            p = Point(r, c)
            if not board.is_on_grid(p) or board.get(p) is not player:
                break
            count += 1
            r += dr
            c += dc

        # Backward
        r, c = move.row - dr, move.col - dc
        while True:
            p = Point(r, c)
            if not board.is_on_grid(p) or board.get(p) is not player:
                break
            count += 1
            r -= dr
            c -= dc

        if count >= WIN_LENGTH:
            return True

    return False


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(game_state: GomokuGameState) -> list[Point]:
    """Return candidate moves near existing stones.

    Priority (highest to lowest):
    1. Empty board → center only.
    2. Opponent has a four → forced block/win squares.
    3. Immediate wins for current player → winning moves only.
    4. Opponent can win next move → blocking moves only.
    5. Opponent has 2+ open-3s → counter-attacks + open-3 blocking squares.
    6. General case: Chebyshev-distance ≤ 2 neighbors (dist-1 first).
    """
    if not game_state.moves:
        center = (BOARD_SIZE + 1) // 2
        return [Point(center, center)]

    board = game_state.board
    current = game_state.current_player
    opponent = current.other

    # Priority 2: forced response to opponent's four
    opp_fours = _four_extension_squares(game_state, opponent)
    if opp_fours:
        my_wins = _four_extension_squares(game_state, current)
        combined = list({
            p for p in (opp_fours + my_wins)
            if board.is_on_grid(p) and board.is_empty(p)
        })
        if combined:
            return combined

    # Priority 3-6: build the general candidate set
    occupied = {m.point for m in game_state.moves}
    dist1: set[Point] = set()
    dist2: set[Point] = set()

    for pt in occupied:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = pt.row + dr, pt.col + dc
                np_ = Point(nr, nc)
                if not board.is_on_grid(np_) or not board.is_empty(np_):
                    continue
                if abs(dr) <= 1 and abs(dc) <= 1:
                    dist1.add(np_)
                else:
                    dist2.add(np_)

    dist2 -= dist1
    candidates = list(dist1) + list(dist2)

    # Priority 3: immediate win
    my_wins = [m for m in candidates if _is_winning_placement(game_state, m, current)]
    if my_wins:
        return my_wins

    # Priority 4: block opponent's immediate win
    opp_wins = [m for m in candidates if _is_winning_placement(game_state, m, opponent)]
    if opp_wins:
        return list({m for m in opp_wins if board.is_empty(m)})

    # Priority 5: double open-3 forced response
    # If opponent has 2+ distinct open-3 patterns they can build a fork next move.
    # Restrict to counter-attacks (creating/blocking an open-four) plus
    # the squares that block the open-3 ends.
    opp_open3_sqrs, opp_open3_count = _open_three_squares(game_state, opponent)
    if opp_open3_count >= 2:
        counter = [m for m in candidates if _move_heuristic(game_state, m) >= 50_000]
        blocks = [p for p in opp_open3_sqrs if board.is_on_grid(p) and board.is_empty(p)]
        combined = list(dict.fromkeys(counter + blocks))
        if len(combined) >= 2:
            return combined

    return candidates


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

def _move_heuristic(game_state: GomokuGameState, move: Point) -> int:
    """Fast heuristic: symmetric offensive + defensive pattern score.

    Uses equal weight (1.0) for both sides to ensure critical defensive
    moves are not undervalued during ordering.
    """
    board = game_state.board
    current = game_state.current_player
    opponent = current.other
    score = 0

    for dr, dc in DIRECTIONS:
        for player in (current, opponent):
            count = 1
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
            fwd = Point(cr, cc)
            if board.is_on_grid(fwd) and board.get(fwd) is None:
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
            bwd = Point(cr, cc)
            if board.is_on_grid(bwd) and board.get(bwd) is None:
                open_ends += 1

            score += _pattern_score(count, open_ends)

    return score


def _sort_key(
    move: Point,
    game_state: GomokuGameState,
    tt_move: Optional[Point],
    killer_set: frozenset,
    history: dict[Point, int],
) -> tuple:
    """Tiered sort key (ascending = better move tried first).

    Tier 0: TT move (from transposition table — best move at this position)
    Tier 1: High-score heuristic ≥ 100_000 (winning or blocking 5-in-a-row)
    Tier 2: Medium-score heuristic ≥ 6_000 (open-4, closed-4, open-3)
    Tier 3: Killer moves (caused beta cutoffs at this depth in sibling nodes)
    Tier 4: All others, ranked by history + heuristic
    """
    if move == tt_move:
        return (0,)
    h = _move_heuristic(game_state, move)
    if h >= 100_000:
        return (1, -h)
    if h >= 6_000:
        return (2, -h)
    if move in killer_set:
        return (3, -history.get(move, 0))
    return (4, -(h + history.get(move, 0)))


def order_moves(
    game_state: GomokuGameState,
    candidates: list[Point],
    tt_move: Optional[Point],
    killers: list[Optional[Point]],
    history: dict[Point, int],
) -> list[Point]:
    """Sort candidates by tiered priority (best first)."""
    killer_set = frozenset(k for k in killers if k is not None)
    return sorted(
        candidates,
        key=lambda m: _sort_key(m, game_state, tt_move, killer_set, history),
    )


# ---------------------------------------------------------------------------
# Quiescence search
# ---------------------------------------------------------------------------

def _quiescence(
    game_state: GomokuGameState,
    alpha: int,
    beta: int,
    color: int,
    qdepth: int,
) -> int:
    """Quiescence search: extend past depth=0 using only forcing moves.

    Avoids the horizon effect where a decisive threat lurks just beyond the
    main search depth.  Uses stand-pat pruning (static eval as lower bound)
    and searches only moves with heuristic >= FORCING_MOVE_HEURISTIC or
    immediate wins/blocks.

    color is +1 for BLACK's turn, -1 for WHITE's turn.
    """
    if game_state.is_over:
        return color * evaluate(game_state)

    stand_pat = color * evaluate(game_state)

    if stand_pat >= beta:
        return stand_pat

    if qdepth <= 0:
        return stand_pat

    alpha = max(alpha, stand_pat)

    candidates = generate_candidates(game_state)
    current = game_state.current_player

    # Collect forcing moves: immediate wins first, then four-or-better threats
    wins: list[Point] = []
    threats: list[Point] = []
    for m in candidates:
        if _is_winning_placement(game_state, m, current):
            wins.append(m)
        elif _move_heuristic(game_state, m) >= FORCING_MOVE_HEURISTIC:
            threats.append(m)

    forcing = (wins + threats)[:MAX_QUIESCENCE_FORCING]
    if not forcing:
        return stand_pat  # quiet position — no threats to resolve

    for move in forcing:
        game_state.apply_move(move)
        score = -_quiescence(game_state, -beta, -alpha, -color, qdepth - 1)
        game_state.undo_move()

        if score >= beta:
            return score
        alpha = max(alpha, score)

    return alpha


# ---------------------------------------------------------------------------
# PVS (Principal Variation Search) inner search
# ---------------------------------------------------------------------------

def _pvs(
    game_state: GomokuGameState,
    depth: int,
    alpha: int,
    beta: int,
    color: int,
    tt: dict,
    killers: list[list[Optional[Point]]],
    history: dict[Point, int],
    zobrist_hash: int,
) -> int:
    """Principal Variation Search with alpha-beta, killer moves, history heuristic,
    LMR, and quiescence search at leaf nodes.

    color is +1 for BLACK's turn, -1 for WHITE's turn.
    Returns the best score from the current player's perspective.
    The Zobrist hash is maintained incrementally by the caller.
    """
    if game_state.is_over:
        return color * evaluate(game_state)

    # Quiescence search at depth=0 instead of bare static eval
    if depth == 0:
        return _quiescence(game_state, alpha, beta, color, QUIESCENCE_DEPTH)

    # Transposition table lookup
    entry = tt.get(zobrist_hash)
    tt_move: Optional[Point] = None
    if entry is not None:
        tt_depth, tt_flag, tt_score, tt_best = entry
        if tt_depth >= depth:
            if tt_flag == 0:                           # exact score
                return tt_score
            elif tt_flag == 1 and tt_score >= beta:    # lower bound — prune
                return tt_score
            elif tt_flag == -1 and tt_score <= alpha:  # upper bound — prune
                return tt_score
        tt_move = tt_best  # use stored best move for ordering even if score unusable

    candidates = generate_candidates(game_state)
    if not candidates:
        return color * evaluate(game_state)

    depth_killers: list[Optional[Point]] = killers[depth] if depth < len(killers) else [None, None]
    valid_killers = [k for k in depth_killers if k is not None and game_state.board.is_empty(k)]
    killer_set = frozenset(valid_killers)

    ordered = order_moves(game_state, candidates, tt_move, valid_killers, history)
    if len(ordered) > MAX_CANDIDATES_INNER:
        forcing_moves = [m for m in ordered if _move_heuristic(game_state, m) >= FORCING_MOVE_HEURISTIC]
        capped = ordered[:MAX_CANDIDATES_INNER]
        ordered = list(dict.fromkeys(forcing_moves + capped))

    # Precompute heuristics for LMR decisions (only at depths where LMR applies)
    if depth >= LMR_MIN_DEPTH and len(ordered) > LMR_MIN_INDEX:
        lmr_heuristics: dict[Point, int] = {
            m: _move_heuristic(game_state, m)
            for m in ordered[LMR_MIN_INDEX:]
        }
    else:
        lmr_heuristics: dict[Point, int] = {}

    orig_alpha = alpha
    best = -INF
    best_move: Optional[Point] = None

    for i, move in enumerate(ordered):
        # Incrementally update hash: remove side-to-move, place stone, flip side
        new_hash = (
            zobrist_hash
            ^ ZOBRIST[move.row][move.col][game_state.current_player.value]
            ^ ZOBRIST_SIDE
        )
        game_state.apply_move(move)

        if i == 0:
            # First move: search with full window (assumed to be PV move)
            score = -_pvs(game_state, depth - 1, -beta, -alpha, -color,
                          tt, killers, history, new_hash)
        else:
            # LMR: reduce depth for late, non-forcing, non-killer quiet moves
            apply_lmr = (
                i >= LMR_MIN_INDEX
                and depth >= LMR_MIN_DEPTH
                and move != tt_move
                and move not in killer_set
                and lmr_heuristics.get(move, LMR_QUIET_THRESHOLD) < LMR_QUIET_THRESHOLD
            )
            if apply_lmr:
                # Reduced-depth null window search
                score = -_pvs(game_state, depth - 2, -alpha - 1, -alpha, -color,
                              tt, killers, history, new_hash)
                if score > alpha:
                    # LMR failed high: re-search at full depth with null window
                    score = -_pvs(game_state, depth - 1, -alpha - 1, -alpha, -color,
                                  tt, killers, history, new_hash)
            else:
                # Standard PVS null window on non-first moves
                score = -_pvs(game_state, depth - 1, -alpha - 1, -alpha, -color,
                              tt, killers, history, new_hash)

            if alpha < score < beta:
                # Null window failed high: re-search with full window
                score = -_pvs(game_state, depth - 1, -beta, -alpha, -color,
                              tt, killers, history, new_hash)

        game_state.undo_move()

        if score > best:
            best = score
            best_move = move
        alpha = max(alpha, score)

        if alpha >= beta:
            # Beta cutoff: update killers and history heuristic
            if depth < len(killers):
                if move != killers[depth][0]:
                    killers[depth][1] = killers[depth][0]
                    killers[depth][0] = move
            history[move] = history.get(move, 0) + (1 << depth)
            break

    # Store result in transposition table
    if best <= orig_alpha:
        flag = -1   # upper bound (we couldn't improve alpha)
    elif best >= beta:
        flag = 1    # lower bound (caused cutoff)
    else:
        flag = 0    # exact score
    tt[zobrist_hash] = (depth, flag, best, best_move)

    return best


# ---------------------------------------------------------------------------
# Root search (one depth iteration)
# ---------------------------------------------------------------------------

def _root_search(
    game_state: GomokuGameState,
    candidates: list[Point],
    depth: int,
    alpha: int,
    beta: int,
    color: int,
    tt: dict,
    killers: list[list[Optional[Point]]],
    history: dict[Point, int],
    zobrist_root: int,
) -> tuple[int, Optional[Point]]:
    """Search all root candidates with PVS. Returns (best_score, best_move)."""
    root_entry = tt.get(zobrist_root)
    tt_move: Optional[Point] = root_entry[3] if root_entry is not None else None
    depth_killers: list[Optional[Point]] = killers[depth] if depth < len(killers) else [None, None]

    ordered = order_moves(game_state, candidates, tt_move, depth_killers, history)
    if len(ordered) > MAX_CANDIDATES_ROOT:
        forcing_moves = [m for m in ordered if _move_heuristic(game_state, m) >= FORCING_MOVE_HEURISTIC]
        capped = ordered[:MAX_CANDIDATES_ROOT]
        ordered = list(dict.fromkeys(forcing_moves + capped))

    best_score = -INF
    best_move: Optional[Point] = ordered[0] if ordered else None

    for i, move in enumerate(ordered):
        new_hash = (
            zobrist_root
            ^ ZOBRIST[move.row][move.col][game_state.current_player.value]
            ^ ZOBRIST_SIDE
        )
        game_state.apply_move(move)

        if i == 0:
            score = -_pvs(game_state, depth - 1, -beta, -alpha, -color,
                          tt, killers, history, new_hash)
        else:
            score = -_pvs(game_state, depth - 1, -alpha - 1, -alpha, -color,
                          tt, killers, history, new_hash)
            if alpha < score < beta:
                score = -_pvs(game_state, depth - 1, -beta, -alpha, -color,
                              tt, killers, history, new_hash)

        game_state.undo_move()

        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best_score, best_move


# ---------------------------------------------------------------------------
# AdvancedAgent
# ---------------------------------------------------------------------------

class AdvancedAgent(Agent):
    """Optimized alpha-beta agent with iterative deepening, PVS, Zobrist TT,
    killer moves, history heuristic, fork detection, quiescence search, LMR,
    broken-four evaluation, and open-3 forced response. Default depth = 6.

    Search algorithm summary:
    - Iterative deepening: depths 1 .. self.depth, reusing TT across iterations
    - Aspiration windows: start with score ± WINDOW; widen on fail-low/high
    - PVS: full window on first move, null window on rest, re-search if needed
    - LMR: reduce depth-1 for late quiet moves, re-search at full depth if needed
    - Killer moves: 2 slots per depth for moves causing recent beta cutoffs
    - History heuristic: global table of move effectiveness, tiebreaker in ordering
    - Forced response: when opponent has a four or double open-3, restrict candidates
    - Fork bonus: evaluation reward for double-threat positions
    - Quiescence search: continue forcing moves beyond depth=0 to avoid horizon
    - Broken-four eval: score non-contiguous 4-in-5-window patterns statically
    """

    def __init__(self, depth: int = 6) -> None:
        self.depth = depth

    @property
    def name(self) -> str:
        return f"AdvancedAgent(d={self.depth})"

    def select_move(self, game_state: GomokuGameState) -> Point:
        color = 1 if game_state.current_player is Player.BLACK else -1
        candidates = generate_candidates(game_state)

        # Fast path: win in 1
        for move in candidates:
            game_state.apply_move(move)
            won = game_state.is_over and game_state.winner is game_state.current_player.other
            game_state.undo_move()
            if won:
                return move

        # Cap root candidates before iterative deepening
        if len(candidates) > MAX_CANDIDATES_ROOT:
            initial_order = order_moves(game_state, candidates, None, [None, None], {})
            forcing_moves = [
                m for m in initial_order
                if _move_heuristic(game_state, m) >= FORCING_MOVE_HEURISTIC
            ]
            capped = initial_order[:MAX_CANDIDATES_ROOT]
            candidates = list(dict.fromkeys(forcing_moves + capped))

        tt: dict = {}
        history: dict[Point, int] = {}
        killers: list[list[Optional[Point]]] = [[None, None] for _ in range(self.depth + 2)]
        best_move: Optional[Point] = candidates[0] if candidates else None
        prev_score = 0
        zobrist_root = _compute_hash(game_state)

        for cur_depth in range(1, self.depth + 1):
            if cur_depth <= 2:
                # No aspiration window on shallow depths (scores are noisy)
                score, move = _root_search(
                    game_state, candidates, cur_depth,
                    -INF, INF, color, tt, killers, history, zobrist_root,
                )
            else:
                # Try narrow aspiration window first
                a = prev_score - ASPIRATION_WINDOW
                b = prev_score + ASPIRATION_WINDOW
                score, move = _root_search(
                    game_state, candidates, cur_depth,
                    a, b, color, tt, killers, history, zobrist_root,
                )
                # Widen to full window on fail-low or fail-high
                if score <= a or score >= b:
                    score, move = _root_search(
                        game_state, candidates, cur_depth,
                        -INF, INF, color, tt, killers, history, zobrist_root,
                    )

            if move is not None:
                best_move = move
            prev_score = score

        assert best_move is not None, "No candidates found"
        return best_move
