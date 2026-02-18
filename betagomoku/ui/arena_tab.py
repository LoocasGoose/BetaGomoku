"""Arena tab: AI vs AI with live board updates."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Generator, Optional

import gradio as gr

from betagomoku.agent.base import Agent
from betagomoku.agent.baseline_agent import BaselineAgent, evaluate
from betagomoku.agent.random_agent import RandomAgent
from betagomoku.game.board import GomokuGameState, format_point
from betagomoku.game.record import save_game
from betagomoku.game.types import Player
from betagomoku.ui.board_component import render_board_svg

ARENA_AGENTS: dict[str, Agent] = {
    "BaselineAgent (d=1)": BaselineAgent(depth=1),
    "BaselineAgent (d=2)": BaselineAgent(depth=2),
    "BaselineAgent (d=3)": BaselineAgent(depth=3),
    "BaselineAgent (d=4)": BaselineAgent(depth=4),
    "BaselineAgent (d=5)": BaselineAgent(depth=5),
    "BaselineAgent (d=6)": BaselineAgent(depth=6),
    "RandomAgent": RandomAgent(),
}

# Ordered list for round-robin grid (exclude RandomAgent)
AGENT_NAMES = [n for n in ARENA_AGENTS if n != "RandomAgent"]

# Short names for the round-robin grid display
SHORT_NAMES: dict[str, str] = {
    name: name.replace("BaselineAgent ", "BA").replace("RandomAgent", "RA")
    for name in AGENT_NAMES
}

MOVE_DELAY = 0.4  # seconds between moves


def _render_arena_board(game: GomokuGameState, result_msg: str = "") -> str:
    eval_score = evaluate(game) if game.moves else None
    return render_board_svg(game, clickable=False, game_over_message=result_msg, eval_score=eval_score)


def _result_message(game: GomokuGameState) -> str:
    if not game.is_over:
        return ""
    if game.winner is Player.BLACK:
        return "Black wins!"
    elif game.winner is Player.WHITE:
        return "White wins!"
    return "Draw!"


def _move_table(game: GomokuGameState) -> list[list[str]]:
    rows: list[list[str]] = []
    for i, move in enumerate(game.moves):
        rows.append([str(i + 1), str(move.player), format_point(move.point)])
    return rows


def _run_arena(
    black_name: str,
    white_name: str,
    delay: float,
    arena_state: dict,
) -> Generator:
    """Generator that yields board updates after each move."""
    black_agent = ARENA_AGENTS.get(black_name, RandomAgent())
    white_agent = ARENA_AGENTS.get(white_name, RandomAgent())
    game = GomokuGameState()

    arena_state["game"] = game
    arena_state["black_name"] = black_name
    arena_state["white_name"] = white_name

    # Initial state
    yield (
        _render_arena_board(game),
        f"Game started: {black_name} (Black) vs {white_name} (White)",
        _move_table(game),
        arena_state,
    )

    while not game.is_over:
        agent = black_agent if game.current_player is Player.BLACK else white_agent
        agent_name = black_name if game.current_player is Player.BLACK else white_name

        move = agent.select_move(game)
        game.apply_move(move)

        result = _result_message(game)
        if result:
            status = f"Game over — {result} ({len(game.moves)} moves)"
        else:
            next_name = black_name if game.current_player is Player.BLACK else white_name
            status = f"Move {len(game.moves)}: {agent_name} played {format_point(move)} — {next_name}'s turn"

        yield (
            _render_arena_board(game, result),
            status,
            _move_table(game),
            arena_state,
        )

        if not game.is_over:
            time.sleep(delay)


def _save_arena_game(arena_state: dict) -> str:
    """Save the most recent arena game."""
    game = arena_state.get("game")
    if game is None or not game.moves:
        return "No game to save. Run a match first."
    filename = save_game(
        game,
        arena_state.get("black_name", "Black"),
        arena_state.get("white_name", "White"),
    )
    return f"Saved: {filename}"


# ---------------------------------------------------------------------------
# Round-robin tournament
# ---------------------------------------------------------------------------

def _make_agent(name: str) -> Agent:
    """Create a fresh agent instance from its name (picklable for multiprocessing)."""
    if name == "RandomAgent":
        return RandomAgent()
    # Parse depth from "BaselineAgent (d=N)"
    depth = int(name.split("=")[1].rstrip(")"))
    return BaselineAgent(depth=depth)


def _play_one_game(black_name: str, white_name: str) -> tuple[str, str, Optional[str]]:
    """Play a full game between two agents. Returns (black_name, white_name, winner_name or None for draw)."""
    black_agent = _make_agent(black_name)
    white_agent = _make_agent(white_name)
    game = GomokuGameState()

    while not game.is_over:
        agent = black_agent if game.current_player is Player.BLACK else white_agent
        move = agent.select_move(game)
        game.apply_move(move)

    if game.winner is Player.BLACK:
        return (black_name, white_name, black_name)
    elif game.winner is Player.WHITE:
        return (black_name, white_name, white_name)
    return (black_name, white_name, None)


def _build_grid(results: dict[tuple[str, str], Optional[str]]) -> list[list[str]]:
    """Build the results grid from completed games. Row = Black, Col = White."""
    grid: list[list[str]] = []
    for row_agent in AGENT_NAMES:
        row = [SHORT_NAMES[row_agent]]
        for col_agent in AGENT_NAMES:
            if row_agent == col_agent:
                row.append("—")
            else:
                key = (row_agent, col_agent)
                if key not in results:
                    row.append("...")
                else:
                    winner = results[key]
                    if winner == row_agent:
                        row.append("W")
                    elif winner is None:
                        row.append("D")
                    else:
                        row.append("L")
        grid.append(row)
    return grid


def _run_round_robin() -> Generator:
    """Run all-vs-all tournament with parallel game execution."""
    # Build matchup list: each pair plays once as each color
    matchups: list[tuple[str, str]] = []
    for i, a in enumerate(AGENT_NAMES):
        for j, b in enumerate(AGENT_NAMES):
            if i != j:
                matchups.append((a, b))

    total = len(matchups)
    results: dict[tuple[str, str], Optional[str]] = {}

    yield (
        f"Starting round-robin: {total} games across {len(AGENT_NAMES)} agents...",
        _build_grid(results),
    )

    completed = 0
    with ProcessPoolExecutor() as pool:
        future_to_match = {
            pool.submit(_play_one_game, black, white): (black, white)
            for black, white in matchups
        }

        for future in as_completed(future_to_match):
            black_name, white_name, winner = future.result()
            results[(black_name, white_name)] = winner
            completed += 1

            # Yield progress update
            if winner is None:
                result_str = "Draw"
            elif winner == black_name:
                result_str = f"{black_name} (B) won"
            else:
                result_str = f"{white_name} (W) won"

            yield (
                f"Game {completed}/{total}: {black_name} vs {white_name} → {result_str}",
                _build_grid(results),
            )

    # Final summary: compute win counts
    wins: dict[str, int] = {name: 0 for name in AGENT_NAMES}
    draws: dict[str, int] = {name: 0 for name in AGENT_NAMES}
    for (black, white), winner in results.items():
        if winner is not None:
            wins[winner] += 1
        else:
            draws[black] += 1
            draws[white] += 1

    ranking = sorted(AGENT_NAMES, key=lambda n: (wins[n], draws[n]), reverse=True)
    summary_lines = [f"Tournament complete! ({total} games)"]
    for i, name in enumerate(ranking):
        losses = (len(AGENT_NAMES) - 1) * 2 - wins[name] - draws[name]
        summary_lines.append(f"  {i+1}. {name}: {wins[name]}W {draws[name]}D {losses}L")

    yield (
        "\n".join(summary_lines),
        _build_grid(results),
    )


def build_arena_tab() -> None:
    """Construct the Arena tab UI inside a gr.Blocks context."""

    arena_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=3):
            board_html = gr.HTML(
                value=_render_arena_board(GomokuGameState()),
                label="Board",
            )
        with gr.Column(scale=1):
            status_text = gr.Textbox(
                value="Select two agents and click Start.",
                label="Status",
                interactive=False,
                lines=2,
            )

            gr.Markdown("### Setup")
            black_choice = gr.Dropdown(
                choices=list(ARENA_AGENTS.keys()),
                value="BaselineAgent (d=2)",
                label="Black Agent",
            )
            white_choice = gr.Dropdown(
                choices=list(ARENA_AGENTS.keys()),
                value="RandomAgent",
                label="White Agent",
            )
            delay_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=MOVE_DELAY,
                step=0.1,
                label="Delay between moves (sec)",
            )
            start_btn = gr.Button("Start", variant="primary")
            save_btn = gr.Button("Save Game")
            save_status = gr.Textbox(label="Save", interactive=False, lines=1)

            gr.Markdown("### Move History")
            move_table = gr.Dataframe(
                headers=["#", "Player", "Move"],
                datatype=["number", "str", "str"],
                interactive=False,
                column_count=3,
            )

    # Round-robin section
    gr.Markdown("---")
    gr.Markdown("### Round Robin Tournament")
    gr.Markdown("Each agent plays every other agent twice (once as Black, once as White).")
    round_robin_btn = gr.Button("All vs All", variant="primary")
    rr_status = gr.Textbox(
        value="Click 'All vs All' to start.",
        label="Tournament Status",
        interactive=False,
        lines=10,
    )
    rr_grid = gr.Dataframe(
        headers=["Black↓ / Opp→"] + [SHORT_NAMES[n] for n in AGENT_NAMES],
        interactive=False,
        column_count=len(AGENT_NAMES) + 1,
    )

    start_btn.click(
        fn=_run_arena,
        inputs=[black_choice, white_choice, delay_slider, arena_state],
        outputs=[board_html, status_text, move_table, arena_state],
    )

    save_btn.click(
        fn=_save_arena_game,
        inputs=[arena_state],
        outputs=[save_status],
    )

    round_robin_btn.click(
        fn=_run_round_robin,
        outputs=[rr_status, rr_grid],
    )
