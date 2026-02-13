from betagomoku.game.types import Player, Point


def test_player_other():
    assert Player.BLACK.other is Player.WHITE
    assert Player.WHITE.other is Player.BLACK


def test_player_str():
    assert str(Player.BLACK) == "Black"
    assert str(Player.WHITE) == "White"


def test_point_is_namedtuple():
    p = Point(3, 5)
    assert p.row == 3
    assert p.col == 5
    assert p == Point(3, 5)
