"""BetaGomoku — Gradio web app entry point."""

import gradio as gr

from betagomoku.ui.arena_tab import build_arena_tab
from betagomoku.ui.board_component import BOARD_CLICK_JS
from betagomoku.ui.play_tab import build_play_tab

with gr.Blocks(title="BetaGomoku") as demo:
    gr.Markdown("# BetaGomoku")
    gr.Markdown("AlphaZero-style Gomoku — 15x15 board, 5 in a row to win.")

    with gr.Tab("Play"):
        build_play_tab()

    with gr.Tab("Arena"):
        build_arena_tab()

    # Placeholder tabs for future phases
    with gr.Tab("Training Dashboard"):
        gr.Markdown("*Coming soon — training metrics and plots.*")

    with gr.Tab("Replay"):
        gr.Markdown("*Coming soon — step through self-play games.*")

    # Bind board click handler JS on page load
    demo.load(fn=None, js=BOARD_CLICK_JS)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
