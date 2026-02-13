# BetaGomoku

AlphaGo-style reinforcement learning agent for Gomoku (9x9 board, 5-in-a-row), with a Gradio web UI.

## Current State

**Phases 1-2 complete**: game engine and interactive Play tab vs a random agent.

- 9x9 board with full game logic (placement, win detection in all directions, undo, draw)
- Clickable SVG board rendered via `gr.HTML` with coordinate text input as fallback
- Play as Black, White, or Random — AI plays the opening move when human is White
- Undo (removes human + AI move pair), resign, new game controls
- Game-over banner overlay on the board (green for win, red for loss, white for draw)

## Setup

```bash
conda create -n betagomoku python=3.12 -y
conda activate betagomoku
pip install -r requirements.txt
```

## Run

```bash
python app.py
# Opens at http://localhost:7860
```

## Test

```bash
python -m pytest tests/ -v
```

## Project Structure

```
betagomoku/
├── game/
│   ├── types.py          # Player enum, Point namedtuple
│   └── board.py          # Board, GomokuGameState, coordinate parsing
├── agent/
│   ├── base.py           # Agent ABC
│   └── random_agent.py   # Random baseline
└── ui/
    ├── board_component.py  # SVG renderer + JS click handler
    └── play_tab.py         # Play tab UI + GameSession
app.py                      # Gradio entry point
tests/                      # Unit tests (39 passing)
```

## Roadmap

- **Phase 3**: Training dashboard — loss curves, win rate, ELO plots
- **Phase 4**: Replay tab — step through self-play games
- **Phase 5**: ML integration — PyTorch model, MCTS agent, encoder, checkpoints
