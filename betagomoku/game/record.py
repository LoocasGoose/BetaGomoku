"""Save and load game records as JSON files."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .board import GomokuGameState, format_point, parse_coordinate

SAVED_GAMES_DIR = Path(__file__).resolve().parents[2] / "saved_games"


def _ensure_dir() -> None:
    SAVED_GAMES_DIR.mkdir(exist_ok=True)


def save_game(
    game: GomokuGameState,
    black_name: str,
    white_name: str,
    result: str = "",
) -> str:
    """Save a game to a JSON file. Returns the filename."""
    _ensure_dir()
    if not result:
        if game.is_over:
            if game.winner is not None:
                result = f"{game.winner} wins"
            else:
                result = "Draw"
        else:
            result = "In progress"

    moves = [format_point(m.point) for m in game.moves]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{black_name}_vs_{white_name}.json"
    # Sanitize filename
    filename = filename.replace(" ", "_").replace("(", "").replace(")", "")

    record = {
        "date": datetime.now().isoformat(),
        "black": black_name,
        "white": white_name,
        "result": result,
        "moves": moves,
    }

    filepath = SAVED_GAMES_DIR / filename
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    return filename


def load_game(filename: str) -> dict:
    """Load a game record from a JSON file. Returns the parsed dict."""
    filepath = SAVED_GAMES_DIR / filename
    with open(filepath) as f:
        return json.load(f)


def list_saved_games() -> list[str]:
    """Return sorted list of saved game filenames (newest first)."""
    _ensure_dir()
    files = [f for f in os.listdir(SAVED_GAMES_DIR) if f.endswith(".json")]
    files.sort(reverse=True)
    return files


def replay_to_move(record: dict, move_index: int) -> GomokuGameState:
    """Rebuild a GomokuGameState with moves replayed up to move_index (inclusive).

    move_index = -1 means empty board, 0 means first move, etc.
    """
    game = GomokuGameState()
    moves = record.get("moves", [])
    for i in range(min(move_index + 1, len(moves))):
        point = parse_coordinate(moves[i])
        if point is not None:
            game.apply_move(point)
    return game
