from betagomoku.game.board import GomokuGameState
from betagomoku.game.types import Player, Point
from betagomoku.ui.board_component import render_board_svg


def test_empty_board_svg():
    g = GomokuGameState()
    html = render_board_svg(g)
    assert "<svg" in html
    assert "</svg>" in html
    assert "gomoku-board" in html
    # Should have click targets for all 81 intersections
    assert html.count('class="board-click"') == 81


def test_svg_with_stones():
    g = GomokuGameState()
    g.apply_move(Point(5, 5))  # Black
    g.apply_move(Point(5, 6))  # White
    html = render_board_svg(g)
    # 2 stones placed, so 79 click targets
    assert html.count('class="board-click"') == 79


def test_svg_not_clickable_when_game_over():
    g = GomokuGameState()
    for i in range(4):
        g.apply_move(Point(1, i + 1))
        g.apply_move(Point(2, i + 1))
    g.apply_move(Point(1, 5))  # Black wins
    html = render_board_svg(g)
    assert html.count('class="board-click"') == 0


def test_svg_not_clickable_when_disabled():
    g = GomokuGameState()
    html = render_board_svg(g, clickable=False)
    assert html.count('class="board-click"') == 0


def test_game_over_banner_displayed():
    g = GomokuGameState()
    for i in range(4):
        g.apply_move(Point(1, i + 1))
        g.apply_move(Point(2, i + 1))
    g.apply_move(Point(1, 5))  # Black wins
    html = render_board_svg(g, game_over_message="You win!")
    assert "You win!" in html
    # Green color for win
    assert "#4ADE80" in html


def test_game_over_banner_ai_wins():
    g = GomokuGameState()
    html = render_board_svg(g, game_over_message="AI wins!")
    # Red color for loss
    assert "#F87171" in html


def test_game_over_banner_draw():
    g = GomokuGameState()
    html = render_board_svg(g, game_over_message="Draw!")
    assert "Draw!" in html
    assert "#FFFFFF" in html


def test_inline_script_present():
    g = GomokuGameState()
    html = render_board_svg(g)
    assert "<script>" in html
    assert "board-click" in html
