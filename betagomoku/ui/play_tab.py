"""Play tab: Human vs AI with interactive SVG board."""

from __future__ import annotations

import random as _random
import time as _time
from dataclasses import dataclass, field
from typing import Optional

import gradio as gr

from betagomoku.agent.advanced_agent import AdvancedAgent
from betagomoku.agent.base import Agent
from betagomoku.agent.baseline_agent import BaselineAgent, evaluate
from betagomoku.agent.random_agent import RandomAgent

AGENT_CHOICES: dict[str, Agent] = {
    "BaselineAgent (d=1)": BaselineAgent(depth=1),
    "BaselineAgent (d=2)": BaselineAgent(depth=2),
    "BaselineAgent (d=3)": BaselineAgent(depth=3),
    "BaselineAgent (d=4)": BaselineAgent(depth=4),
    "BaselineAgent (d=5)": BaselineAgent(depth=5),
    "BaselineAgent (d=6)": BaselineAgent(depth=6),
    "BaselineAdvanced (d=6)": AdvancedAgent(depth=6),
    "RandomAgent": RandomAgent(),
}
from betagomoku.game.board import (
    GomokuGameState,
    format_point,
    parse_coordinate,
)
from betagomoku.game.record import save_game
from betagomoku.game.types import Player
from betagomoku.ui.board_component import render_board_svg


@dataclass
class GameSession:
    """Per-tab game state held in gr.State."""

    game: GomokuGameState = field(default_factory=GomokuGameState)
    agent: Agent = field(default_factory=lambda: BaselineAgent(depth=2))
    human_player: Player = field(default=Player.BLACK)
    _turn_start: float = field(default_factory=_time.time)

    def reset(self, human_player: Optional[Player] = None) -> None:
        self.game = GomokuGameState()
        self._turn_start = _time.time()
        if human_player is not None:
            self.human_player = human_player

    def mark_turn_start(self) -> None:
        """Record the moment the current player's clock starts."""
        self._turn_start = _time.time()

    def elapsed_since_turn_start(self) -> float:
        return _time.time() - self._turn_start

    @property
    def game_over_banner(self) -> str:
        """Short text for the SVG overlay banner. Empty if game is not over."""
        g = self.game
        if not g.is_over:
            return ""
        if g.winner is not None:
            if g.winner == self.human_player:
                return "You win!"
            return "AI wins!"
        return "Draw!"

    @property
    def status_text(self) -> str:
        g = self.game
        if g.is_over:
            if g.winner is not None:
                who = "You win!" if g.winner == self.human_player else "AI wins!"
                return f"Game over — {who} ({g.winner} by 5-in-a-row)"
            return "Game over — Draw!"
        if g.current_player == self.human_player:
            return f"Your turn ({g.current_player})"
        return f"AI is thinking... ({g.current_player})"

    @property
    def move_history_table(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for i, move in enumerate(self.game.moves):
            t = f"{move.elapsed:.2f}" if move.elapsed is not None else "—"
            rows.append([str(i + 1), str(move.player), format_point(move.point), t])
        return rows


def _make_board_html(session: GameSession) -> str:
    clickable = (
        not session.game.is_over
        and session.game.current_player == session.human_player
    )
    eval_score = evaluate(session.game) if session.game.moves else None
    return render_board_svg(
        session.game,
        clickable=clickable,
        game_over_message=session.game_over_banner,
        eval_score=eval_score,
    )


def _ai_opening_move(session: GameSession) -> None:
    """If AI goes first (human is White), let the AI play the opening move."""
    if (
        session.human_player == Player.WHITE
        and not session.game.moves
        and not session.game.is_over
    ):
        t0 = _time.time()
        ai_move = session.agent.select_move(session.game)
        session.game.apply_move(ai_move, elapsed=_time.time() - t0)
        session.mark_turn_start()  # human's clock starts now


def _apply_human_move(coord_text: str, session: GameSession):
    """Process a human move, then let the AI respond."""
    if session.game.is_over:
        return (
            _make_board_html(session),
            session.status_text,
            session.move_history_table,
            session,
            "",  # clear coord input
        )

    if session.game.current_player != session.human_player:
        return (
            _make_board_html(session),
            "Wait — it's the AI's turn.",
            session.move_history_table,
            session,
            "",
        )

    point = parse_coordinate(coord_text)
    if point is None:
        return (
            _make_board_html(session),
            f"Invalid coordinate: '{coord_text}'. Use format like E5.",
            session.move_history_table,
            session,
            "",
        )

    if not session.game.board.is_empty(point):
        return (
            _make_board_html(session),
            f"{format_point(point)} is already occupied.",
            session.move_history_table,
            session,
            "",
        )

    # Human move (time since their turn started)
    human_elapsed = session.elapsed_since_turn_start()
    session.game.apply_move(point, elapsed=human_elapsed)

    # AI response (if game isn't over)
    if not session.game.is_over:
        t0 = _time.time()
        ai_move = session.agent.select_move(session.game)
        session.game.apply_move(ai_move, elapsed=_time.time() - t0)
        session.mark_turn_start()  # human's clock starts again

    return (
        _make_board_html(session),
        session.status_text,
        session.move_history_table,
        session,
        "",
    )


def _new_game_with_color(color_choice: str, agent_choice: str, session: GameSession):
    """Start a new game. color_choice is 'Black', 'White', or 'Random'."""
    if color_choice == "Random":
        human = _random.choice([Player.BLACK, Player.WHITE])
    elif color_choice == "White":
        human = Player.WHITE
    else:
        human = Player.BLACK

    session.agent = AGENT_CHOICES.get(agent_choice, RandomAgent())
    session.reset(human_player=human)

    # If human is White, AI (Black) plays first; _ai_opening_move marks turn start after
    # If human is Black, mark turn start now (reset already sets it, but be explicit)
    _ai_opening_move(session)
    if human is Player.BLACK:
        session.mark_turn_start()

    assigned = "Black" if human is Player.BLACK else "White"
    return (
        _make_board_html(session),
        session.status_text,
        session.move_history_table,
        session,
        f"You are {assigned}.",
    )


def _undo_move(session: GameSession):
    """Undo the last move pair (AI + human)."""
    if not session.game.moves:
        return (
            _make_board_html(session),
            "Nothing to undo.",
            session.move_history_table,
            session,
        )

    # If the last move was AI's, undo both AI and human
    last = session.game.moves[-1]
    if last.player != session.human_player:
        session.game.undo_move()  # undo AI
    if session.game.moves:
        session.game.undo_move()  # undo human

    return (
        _make_board_html(session),
        session.status_text,
        session.move_history_table,
        session,
    )


def _resign(session: GameSession):
    if session.game.is_over:
        return (
            _make_board_html(session),
            session.status_text,
            session.move_history_table,
            session,
        )
    session.game._is_over = True
    session.game._winner = session.human_player.other
    return (
        _make_board_html(session),
        session.status_text,
        session.move_history_table,
        session,
    )


def _save_game(session: GameSession) -> str:
    """Save the current game to disk."""
    if not session.game.moves:
        return "No moves to save."
    human_color = "Black" if session.human_player is Player.BLACK else "White"
    agent_name = session.agent.name if hasattr(session.agent, "name") else "AI"
    if session.human_player is Player.BLACK:
        black_name, white_name = f"Human_{human_color}", agent_name
    else:
        black_name, white_name = agent_name, f"Human_{human_color}"
    filename = save_game(session.game, black_name, white_name)
    return f"Saved: {filename}"


def build_play_tab() -> None:
    """Construct the Play tab UI inside a gr.Blocks context."""

    session_state = gr.State(GameSession())

    with gr.Row():
        # Left: board
        with gr.Column(scale=3):
            board_html = gr.HTML(
                value=render_board_svg(GomokuGameState()),
                label="Board",
            )
        # Right: controls
        with gr.Column(scale=1):
            status_text = gr.Textbox(
                value="Your turn (Black)",
                label="Status",
                interactive=False,
                lines=2,
            )
            color_info = gr.Textbox(
                value="You are Black.",
                label="Color",
                interactive=False,
                lines=1,
            )

            gr.Markdown("### New Game")
            color_choice = gr.Radio(
                choices=["Random", "Black", "White"],
                value="Random",
                label="Play as",
            )
            agent_choice = gr.Dropdown(
                choices=list(AGENT_CHOICES.keys()),
                value=list(AGENT_CHOICES.keys())[0],
                label="Opponent",
            )
            new_game_btn = gr.Button("New Game", variant="primary")

            with gr.Row():
                undo_btn = gr.Button("Undo")
                resign_btn = gr.Button("Resign", variant="stop")
            save_btn = gr.Button("Save Game")
            save_status = gr.Textbox(label="Save", interactive=False, lines=1)

            gr.Markdown("### Enter Move")
            coord_input = gr.Textbox(
                label="Coordinate (e.g. E5)",
                placeholder="E5",
                elem_id="coord-input",
                lines=1,
            )
            coord_submit = gr.Button(
                "Submit Move",
                elem_id="coord-submit",
            )

            gr.Markdown("### Move History")
            move_table = gr.Dataframe(
                headers=["#", "Player", "Move", "Time (s)"],
                datatype=["number", "str", "str", "str"],
                interactive=False,
                column_count=4,
            )

    # Outputs shared by most callbacks
    board_outputs = [board_html, status_text, move_table, session_state]

    # Wire up callbacks
    coord_submit.click(
        fn=_apply_human_move,
        inputs=[coord_input, session_state],
        outputs=board_outputs + [coord_input],
    )

    new_game_btn.click(
        fn=_new_game_with_color,
        inputs=[color_choice, agent_choice, session_state],
        outputs=board_outputs + [color_info],
    )

    undo_btn.click(
        fn=_undo_move,
        inputs=[session_state],
        outputs=board_outputs,
    )

    resign_btn.click(
        fn=_resign,
        inputs=[session_state],
        outputs=board_outputs,
    )

    save_btn.click(
        fn=_save_game,
        inputs=[session_state],
        outputs=[save_status],
    )
