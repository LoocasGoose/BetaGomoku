# BetaGomoku — Project Plan & Progress

## Overview
AlphaZero-style RL Gomoku agent (15x15, 5-in-a-row) with Gradio web UI.

## Environment
- **Conda env**: `betagomoku` (Python 3.12), Gradio 6.5.1
- **Run**: `conda activate betagomoku && python app.py` → localhost:7860
- **Tests**: 74 passing — `python -m pytest tests/ -v`

## Completed

### Phase 1-2: Game Engine + Play Tab
- `Point(row, col)` 1-indexed, `Player` enum, `Board` dict-based, `GomokuGameState` with win/draw/undo
- SVG board via `gr.HTML` with JS click handler, `GameSession` in `gr.State`
- Play as Black/White/Random, undo (pair), resign, new game, move history

### Baseline Agent
- Negamax + alpha-beta with pattern-based static evaluation
- Candidate generation (Chebyshev distance 2), move ordering heuristic
- **Optimizations**: transposition table, candidate cap (20), root alpha-beta fix, occupied-only eval scan
- Depths 1-4 available; d=4 runs ~1s per move

### Arena Tab
- AI vs AI with live board updates, configurable delay
- Save completed games to `saved_games/`

### Eval Bar
- Chess.com-style vertical bar beside the board (tanh-mapped, 0-1)
- Shows in Play tab, Arena tab, and Replay tab

### Replay Tab
- Load saved JSON games, step forward/backward (`<< < > >>`), jump to start/end
- Board with eval bar updates at each step, move history table

### Save/Load System
- `betagomoku/game/record.py` — save/load/list JSON game records in `saved_games/`
- Save buttons on both Play and Arena tabs

## Remaining

### Phase 3: Training Dashboard
- `betagomoku/training/metrics.py` — `MetricsStore`: read/append JSONL training logs
- `betagomoku/ui/training_tab.py` — loss curves (policy + value), win rate, ELO plots

### Phase 4: ML Integration
- `betagomoku/agent/encoder.py` — board → 3-plane tensor
- `betagomoku/agent/mcts_agent.py` — MCTS + neural net (AlphaZero-style)
- `betagomoku/training/` — self-play pipeline, checkpoint management
- Wire model/MCTS into Play tab with checkpoint dropdown + sim slider
- **Framework**: PyTorch
