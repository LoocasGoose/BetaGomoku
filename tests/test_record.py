"""Tests for game record save/load functionality."""

import json
import os

import pytest

from betagomoku.game.board import GomokuGameState, format_point
from betagomoku.game.record import (
    SAVED_GAMES_DIR,
    list_saved_games,
    load_game,
    replay_to_move,
    save_game,
)
from betagomoku.game.types import Player, Point


@pytest.fixture
def sample_game():
    """Create a short game with a few moves."""
    game = GomokuGameState()
    game.apply_move(Point(8, 8))   # Black
    game.apply_move(Point(8, 9))   # White
    game.apply_move(Point(7, 7))   # Black
    return game


@pytest.fixture(autouse=True)
def cleanup_saved_files():
    """Clean up any test-generated save files."""
    yield
    # Remove files created during tests (identified by test player names)
    if SAVED_GAMES_DIR.exists():
        for f in os.listdir(SAVED_GAMES_DIR):
            if "TestBlack" in f or "TestWhite" in f:
                os.remove(SAVED_GAMES_DIR / f)


class TestSaveGame:
    def test_save_creates_file(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite")
        assert filename.endswith(".json")
        assert (SAVED_GAMES_DIR / filename).exists()

    def test_save_content(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite")
        with open(SAVED_GAMES_DIR / filename) as f:
            data = json.load(f)
        assert data["black"] == "TestBlack"
        assert data["white"] == "TestWhite"
        assert len(data["moves"]) == 3
        assert data["moves"][0] == "H8"
        assert data["moves"][1] == "I8"
        assert data["moves"][2] == "G7"

    def test_save_auto_result(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite")
        data = load_game(filename)
        assert data["result"] == "In progress"

    def test_save_custom_result(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite", result="Test result")
        data = load_game(filename)
        assert data["result"] == "Test result"


class TestLoadGame:
    def test_load_roundtrip(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite")
        data = load_game(filename)
        assert data["black"] == "TestBlack"
        assert len(data["moves"]) == 3


class TestListSavedGames:
    def test_list_returns_json_files(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite")
        files = list_saved_games()
        assert filename in files

    def test_list_sorted_newest_first(self, sample_game):
        files = list_saved_games()
        # Files are named with timestamps, so reverse sort = newest first
        assert files == sorted(files, reverse=True)


class TestReplayToMove:
    def test_empty_board(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite")
        record = load_game(filename)
        game = replay_to_move(record, -1)
        assert len(game.moves) == 0

    def test_first_move(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite")
        record = load_game(filename)
        game = replay_to_move(record, 0)
        assert len(game.moves) == 1
        assert game.moves[0].point == Point(8, 8)

    def test_all_moves(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite")
        record = load_game(filename)
        game = replay_to_move(record, 2)
        assert len(game.moves) == 3

    def test_beyond_total(self, sample_game):
        filename = save_game(sample_game, "TestBlack", "TestWhite")
        record = load_game(filename)
        game = replay_to_move(record, 100)
        assert len(game.moves) == 3  # capped at total
