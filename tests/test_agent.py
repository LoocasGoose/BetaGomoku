from betagomoku.agent.random_agent import RandomAgent
from betagomoku.game.board import GomokuGameState
from betagomoku.game.types import Point


def test_random_agent_returns_legal_move():
    g = GomokuGameState()
    agent = RandomAgent()
    for _ in range(10):
        move = agent.select_move(g)
        assert isinstance(move, Point)
        assert g.board.is_empty(move)
        g.apply_move(move)


def test_random_agent_name():
    assert RandomAgent().name == "RandomAgent"
