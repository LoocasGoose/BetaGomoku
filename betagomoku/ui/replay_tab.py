"""Replay tab: step through saved games."""

from __future__ import annotations

from dataclasses import dataclass, field

import gradio as gr

from betagomoku.agent.baseline_agent import evaluate
from betagomoku.game.board import GomokuGameState, format_point
from betagomoku.game.record import list_saved_games, load_game, replay_to_move
from betagomoku.ui.board_component import render_board_svg


@dataclass
class ReplayState:
    """Per-tab replay state held in gr.State."""

    record: dict = field(default_factory=dict)
    move_index: int = -1  # -1 = empty board

    @property
    def total_moves(self) -> int:
        return len(self.record.get("moves", []))

    @property
    def status_text(self) -> str:
        if not self.record:
            return "Load a game to begin."
        info = f"{self.record.get('black', '?')} (Black) vs {self.record.get('white', '?')} (White)"
        result = self.record.get("result", "")
        pos = f"Move {self.move_index + 1}/{self.total_moves}" if self.move_index >= 0 else "Start"
        parts = [info, f"Result: {result}", pos]
        return " | ".join(parts)

    @property
    def move_table(self) -> list[list[str]]:
        moves = self.record.get("moves", [])
        rows: list[list[str]] = []
        for i in range(min(self.move_index + 1, len(moves))):
            player = "Black" if i % 2 == 0 else "White"
            rows.append([str(i + 1), player, moves[i]])
        return rows


def _render_replay_board(state: ReplayState) -> str:
    if not state.record:
        return render_board_svg(GomokuGameState(), clickable=False)
    game = replay_to_move(state.record, state.move_index)
    eval_score = evaluate(game) if game.moves else None
    game_over_msg = ""
    if state.move_index + 1 >= state.total_moves and game.is_over:
        winner = state.record.get("result", "")
        game_over_msg = winner
    return render_board_svg(
        game, clickable=False, eval_score=eval_score,
        game_over_message=game_over_msg,
    )


def _load_game(filename: str, state: ReplayState):
    """Load a saved game file."""
    if not filename:
        return (
            _render_replay_board(state),
            "Select a game file.",
            [],
            state,
        )
    state.record = load_game(filename)
    state.move_index = -1
    return (
        _render_replay_board(state),
        state.status_text,
        state.move_table,
        state,
    )


def _step_forward(state: ReplayState):
    if not state.record:
        return _render_replay_board(state), state.status_text, state.move_table, state
    if state.move_index < state.total_moves - 1:
        state.move_index += 1
    return _render_replay_board(state), state.status_text, state.move_table, state


def _step_backward(state: ReplayState):
    if not state.record:
        return _render_replay_board(state), state.status_text, state.move_table, state
    if state.move_index >= 0:
        state.move_index -= 1
    return _render_replay_board(state), state.status_text, state.move_table, state


def _jump_start(state: ReplayState):
    if not state.record:
        return _render_replay_board(state), state.status_text, state.move_table, state
    state.move_index = -1
    return _render_replay_board(state), state.status_text, state.move_table, state


def _jump_end(state: ReplayState):
    if not state.record:
        return _render_replay_board(state), state.status_text, state.move_table, state
    state.move_index = state.total_moves - 1
    return _render_replay_board(state), state.status_text, state.move_table, state


def _refresh_file_list():
    files = list_saved_games()
    return gr.update(choices=files, value=files[0] if files else None)


def build_replay_tab() -> None:
    """Construct the Replay tab UI inside a gr.Blocks context."""

    replay_state = gr.State(ReplayState())

    with gr.Row():
        with gr.Column(scale=3):
            board_html = gr.HTML(
                value=render_board_svg(GomokuGameState(), clickable=False),
                label="Board",
            )
        with gr.Column(scale=1):
            status_text = gr.Textbox(
                value="Load a game to begin.",
                label="Status",
                interactive=False,
                lines=2,
            )

            gr.Markdown("### Load Game")
            file_dropdown = gr.Dropdown(
                choices=list_saved_games(),
                label="Saved Games",
            )
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                load_btn = gr.Button("Load", variant="primary")

            gr.Markdown("### Controls")
            with gr.Row():
                start_btn = gr.Button("<<")
                back_btn = gr.Button("<")
                fwd_btn = gr.Button(">")
                end_btn = gr.Button(">>")

            gr.Markdown("### Move History")
            move_table = gr.Dataframe(
                headers=["#", "Player", "Move"],
                datatype=["number", "str", "str"],
                interactive=False,
                column_count=3,
            )

    outputs = [board_html, status_text, move_table, replay_state]

    load_btn.click(
        fn=_load_game,
        inputs=[file_dropdown, replay_state],
        outputs=outputs,
    )

    refresh_btn.click(
        fn=_refresh_file_list,
        outputs=[file_dropdown],
    )

    fwd_btn.click(fn=_step_forward, inputs=[replay_state], outputs=outputs)
    back_btn.click(fn=_step_backward, inputs=[replay_state], outputs=outputs)
    start_btn.click(fn=_jump_start, inputs=[replay_state], outputs=outputs)
    end_btn.click(fn=_jump_end, inputs=[replay_state], outputs=outputs)
