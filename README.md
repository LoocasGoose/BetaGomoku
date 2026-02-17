# BetaGomoku

AlphaZero-style Gomoku agent with a Gradio web UI. 15x15 board, 5-in-a-row to win.

## Features

- **Play tab** — Human vs AI with clickable SVG board, undo, resign, eval bar
- **Arena tab** — AI vs AI with live board updates and eval bar
- **Replay tab** — Load and step through saved games
- **Baseline agent** — Negamax + alpha-beta with pattern eval, transposition table (depths 1-4)
- **Save/load** — Save any game to JSON, replay later

## Setup

```bash
conda create -n betagomoku python=3.12 -y
conda activate betagomoku
pip install -r requirements.txt
python app.py  # http://localhost:7860
```

## Test

```bash
python -m pytest tests/ -v  # 74 tests
```

## Structure

```
betagomoku/
├── game/
│   ├── types.py            # Player enum, Point namedtuple
│   ├── board.py            # Board, GomokuGameState, coordinate parsing
│   └── record.py           # Save/load game records (JSON)
├── agent/
│   ├── base.py             # Agent ABC
│   ├── random_agent.py     # Random baseline
│   └── baseline_agent.py   # Negamax + alpha-beta + transposition table
└── ui/
    ├── board_component.py  # SVG renderer + eval bar + JS click handler
    ├── play_tab.py         # Human vs AI
    ├── arena_tab.py        # AI vs AI
    └── replay_tab.py       # Step through saved games
```

## Roadmap

- **Phase 3**: Training dashboard — loss curves, win rate, ELO plots
- **Phase 4**: ML integration — PyTorch model, MCTS agent, self-play pipeline
