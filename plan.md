# BetaGomoku — Project Plan & Progress

## Overview
AlphaZero-style RL Gomoku agent with a Gradio-based web UI for interactive play and training monitoring.

## Architecture
- **Game engine**: `betagomoku/game/` — types, board logic, rendering
- **Agents**: `betagomoku/agent/` — agent ABC, random baseline, future MCTS+neural net
- **UI**: `betagomoku/ui/` — Gradio tabs (play, training dashboard, replay)
- **Training**: `betagomoku/training/` — metrics store, checkpoint manager (future)
- **Entry point**: `app.py` — Gradio app with tabbed interface
- **Reference only**: `alphago/` — not imported by betagomoku

## Environment
- **Conda env**: `betagomoku` (Python 3.12)
- **Dependencies**: `gradio>=4.0` (currently running Gradio 6.5.1), `pytest`
- **Run**: `conda activate betagomoku && python app.py` → localhost:7860

## Completed — Phase 1 & 2 (Game Engine + Play Tab)

### Files created

| File | Purpose |
|------|---------|
| `betagomoku/game/types.py` | `Player` enum (BLACK/WHITE with `.other`), `Point` namedtuple (row, col, 1-indexed) |
| `betagomoku/game/board.py` | `Board` (place/remove/get), `GomokuGameState` (9x9, 5-in-a-row win detection, undo, draw detection), `parse_coordinate`/`format_point` helpers |
| `betagomoku/agent/base.py` | `Agent` ABC with `select_move(game_state) -> Point` |
| `betagomoku/agent/random_agent.py` | `RandomAgent` — picks random legal move |
| `betagomoku/ui/board_component.py` | SVG board renderer (clickable intersections, stone placement, last-move marker, coordinate labels) + JS click handler that writes to hidden Gradio textbox |
| `betagomoku/ui/play_tab.py` | `GameSession` dataclass (held in `gr.State`), `build_play_tab()` with: SVG board, status text, New Game / Undo / Resign buttons, coordinate text input, move history table |
| `app.py` | Gradio entry point — Play tab active, Training Dashboard & Replay as placeholders |
| `requirements.txt` | `gradio>=4.0` |
| `tests/test_types.py` | Player enum and Point tests |
| `tests/test_board.py` | Board, GameState, coordinate parsing tests (21 tests covering wins in all 4 directions, undo, draw, error cases) |
| `tests/test_agent.py` | RandomAgent legality and naming tests |
| `tests/test_renderer.py` | SVG output structure, click target counts, game-over behavior |

### Key design decisions
1. **SVG via `gr.HTML`** — clickable circle elements with `data-coord` attributes; JS click handler sets a hidden textbox value and triggers submit button
2. **`gr.State` holds `GameSession`** — game state, agent, and human player color per browser tab
3. **1-indexed coordinates** — Point(row=1, col=1) is bottom-left, displayed as "A1"
4. **Undo removes move pairs** — undoes AI move then human move
5. **Gradio 6.x compatible** — theme passed to `launch()`, `column_count` instead of `col_count`

### Test results
28 tests, all passing.

## Completed — Baseline Agent (Negamax + Alpha-Beta)

### Files created/modified

| File | Purpose |
|------|---------|
| `betagomoku/agent/baseline_agent.py` | `BaselineAgent`: negamax search with alpha-beta pruning, pattern-based static evaluation, candidate generation (Chebyshev distance 2), move ordering heuristic |
| `tests/test_baseline_agent.py` | 21 tests: pattern scoring, evaluation, candidate generation, search (wins/blocks), agent config |
| `betagomoku/ui/play_tab.py` | Added opponent dropdown (BaselineAgent d=2, RandomAgent) |

### Key design decisions
1. **Pattern scoring**: consecutive groups scored by (count, open_ends) — five=100k, open-four=10k, etc.
2. **Absolute evaluation**: positive = BLACK advantage; negamax uses color multiplier (+1/-1)
3. **Chebyshev distance 2 candidates**: limits branching factor to ~20-50 moves vs 225 legal moves
4. **Move ordering**: offensive pattern score + 0.5 * defensive (blocking) score for alpha-beta efficiency
5. **Depth 2 default**: safe for real-time play on 15x15
6. **Immediate win check**: early exit before full search if a winning move exists

### Test results
61 tests, all passing.

## Remaining Phases

### Phase 3: Training Dashboard
- `betagomoku/training/metrics.py` — `MetricsStore`: read/append JSONL training logs
- `betagomoku/ui/training_tab.py` — loss curves (policy + value), win rate vs random, ELO rating plots (matplotlib via `gr.Plot`), auto-refresh toggle, MCTS stats table by checkpoint

### Phase 4: Replay Tab
- Game replay JSON format (move list + per-move MCTS stats)
- `betagomoku/ui/replay_tab.py` — SVG board (read-only), game selector dropdown, step controls (|< < > >|), move list sidebar

### Phase 5: ML Integration
- `betagomoku/agent/encoder.py` — board state → 3-plane tensor (my stones, opp stones, turn indicator)
- `betagomoku/agent/mcts_agent.py` — MCTS + neural net (AlphaZero-style)
- `betagomoku/training/checkpoint.py` — list/load local model checkpoints
- Wire checkpoint dropdown + MCTS sim slider into Play tab
- **Framework**: PyTorch
