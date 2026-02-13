from betagomoku.game.types import Player
from betagomoku.ui.play_tab import GameSession, _new_game_with_color


def test_new_game_as_black():
    session = GameSession()
    result = _new_game_with_color("Black", session)
    assert session.human_player is Player.BLACK
    assert len(session.game.moves) == 0  # no AI opening move
    assert "You are Black" in result[4]


def test_new_game_as_white_ai_goes_first():
    session = GameSession()
    result = _new_game_with_color("White", session)
    assert session.human_player is Player.WHITE
    # AI (Black) should have played the opening move
    assert len(session.game.moves) == 1
    assert session.game.moves[0].player is Player.BLACK
    assert session.game.current_player is Player.WHITE
    assert "You are White" in result[4]


def test_new_game_random_assigns_valid_color():
    session = GameSession()
    colors_seen = set()
    for _ in range(50):
        _new_game_with_color("Random", session)
        colors_seen.add(session.human_player)
    # With 50 tries, we should see both colors
    assert Player.BLACK in colors_seen
    assert Player.WHITE in colors_seen


def test_game_over_banner_win():
    session = GameSession()
    session.human_player = Player.BLACK
    session.game._is_over = True
    session.game._winner = Player.BLACK
    assert session.game_over_banner == "You win!"


def test_game_over_banner_loss():
    session = GameSession()
    session.human_player = Player.BLACK
    session.game._is_over = True
    session.game._winner = Player.WHITE
    assert session.game_over_banner == "AI wins!"


def test_game_over_banner_draw():
    session = GameSession()
    session.game._is_over = True
    session.game._winner = None
    assert session.game_over_banner == "Draw!"


def test_game_over_banner_empty_when_playing():
    session = GameSession()
    assert session.game_over_banner == ""
