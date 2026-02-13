from betagomoku.game.board import GomokuGameState
from betagomoku.game.types import Player, Point
from betagomoku.ui.board_component import render_board_svg


def test_empty_board_svg():
    g = GomokuGameState()
    svg = render_board_svg(g)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "gomoku-board" in svg
    # Should have click targets for all 81 intersections
    assert svg.count('class="board-click"') == 81


def test_svg_with_stones():
    g = GomokuGameState()
    g.apply_move(Point(5, 5))  # Black
    g.apply_move(Point(5, 6))  # White
    svg = render_board_svg(g)
    # 2 stones placed, so 79 click targets
    assert svg.count('class="board-click"') == 79


def test_svg_not_clickable_when_game_over():
    g = GomokuGameState()
    for i in range(4):
        g.apply_move(Point(1, i + 1))
        g.apply_move(Point(2, i + 1))
    g.apply_move(Point(1, 5))  # Black wins
    svg = render_board_svg(g)
    assert svg.count('class="board-click"') == 0


def test_svg_not_clickable_when_disabled():
    g = GomokuGameState()
    svg = render_board_svg(g, clickable=False)
    assert svg.count('class="board-click"') == 0
