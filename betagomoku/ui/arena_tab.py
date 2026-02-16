"""Arena tab: AI vs AI with live board updates."""

from __future__ import annotations

import time
from typing import Generator

import gradio as gr

from betagomoku.agent.base import Agent
from betagomoku.agent.baseline_agent import BaselineAgent
from betagomoku.agent.random_agent import RandomAgent
from betagomoku.game.board import GomokuGameState, format_point
from betagomoku.game.types import Player
from betagomoku.ui.board_component import render_board_svg

ARENA_AGENTS: dict[str, Agent] = {
    "BaselineAgent (d=2)": BaselineAgent(depth=2),
    "BaselineAgent (d=3)": BaselineAgent(depth=3),
    "RandomAgent": RandomAgent(),
}

MOVE_DELAY = 0.4  # seconds between moves


def _render_arena_board(game: GomokuGameState, result_msg: str = "") -> str:
    return render_board_svg(game, clickable=False, game_over_message=result_msg)


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
) -> Generator:
    """Generator that yields board updates after each move."""
    black_agent = ARENA_AGENTS.get(black_name, RandomAgent())
    white_agent = ARENA_AGENTS.get(white_name, RandomAgent())
    game = GomokuGameState()

    # Initial state
    yield (
        _render_arena_board(game),
        f"Game started: {black_name} (Black) vs {white_name} (White)",
        _move_table(game),
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
        )

        if not game.is_over:
            time.sleep(delay)


def build_arena_tab() -> None:
    """Construct the Arena tab UI inside a gr.Blocks context."""

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

            gr.Markdown("### Move History")
            move_table = gr.Dataframe(
                headers=["#", "Player", "Move"],
                datatype=["number", "str", "str"],
                interactive=False,
                column_count=3,
            )

    start_btn.click(
        fn=_run_arena,
        inputs=[black_choice, white_choice, delay_slider],
        outputs=[board_html, status_text, move_table],
    )
