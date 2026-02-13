"""SVG board renderer + JavaScript click handler for Gradio."""

from __future__ import annotations

from typing import Optional

from betagomoku.game.board import BOARD_SIZE, COL_LABELS, GomokuGameState
from betagomoku.game.types import Player, Point

# Layout constants
CELL_SIZE = 50
MARGIN = 40
BOARD_PX = MARGIN * 2 + CELL_SIZE * (BOARD_SIZE - 1)
STONE_RADIUS = 20
CLICK_RADIUS = 22  # Invisible click target radius

# Colors
BG_COLOR = "#DCB35C"
LINE_COLOR = "#4A3728"
BLACK_STONE = "#1A1A1A"
WHITE_STONE = "#F5F5F5"
WHITE_STROKE = "#888"


def _coord(row: int, col: int) -> tuple[int, int]:
    """Convert 1-indexed board coordinates to SVG pixel coordinates."""
    x = MARGIN + (col - 1) * CELL_SIZE
    y = MARGIN + (BOARD_SIZE - row) * CELL_SIZE  # row 1 at bottom
    return x, y


def render_board_svg(
    game_state: GomokuGameState,
    clickable: bool = True,
    highlight_last: bool = True,
    game_over_message: str = "",
) -> str:
    """Render the board as an SVG string wrapped in a div with click JS."""
    parts: list[str] = []

    # Wrapper div
    parts.append(f'<div id="gomoku-board-wrap" style="display:inline-block;position:relative;">')

    # SVG header
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{BOARD_PX}" height="{BOARD_PX}" '
        f'viewBox="0 0 {BOARD_PX} {BOARD_PX}" '
        f'id="gomoku-board">'
    )

    # Background
    parts.append(
        f'<rect width="{BOARD_PX}" height="{BOARD_PX}" fill="{BG_COLOR}" rx="4"/>'
    )

    # Grid lines
    for i in range(BOARD_SIZE):
        x = MARGIN + i * CELL_SIZE
        y_top = MARGIN
        y_bot = MARGIN + (BOARD_SIZE - 1) * CELL_SIZE
        parts.append(
            f'<line x1="{x}" y1="{y_top}" x2="{x}" y2="{y_bot}" '
            f'stroke="{LINE_COLOR}" stroke-width="1"/>'
        )
        y = MARGIN + i * CELL_SIZE
        x_left = MARGIN
        x_right = MARGIN + (BOARD_SIZE - 1) * CELL_SIZE
        parts.append(
            f'<line x1="{x_left}" y1="{y}" x2="{x_right}" y2="{y}" '
            f'stroke="{LINE_COLOR}" stroke-width="1"/>'
        )

    # Star points (center for 9x9)
    center = (BOARD_SIZE + 1) // 2
    cx, cy = _coord(center, center)
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{LINE_COLOR}"/>')

    # Column labels (top and bottom)
    for c in range(1, BOARD_SIZE + 1):
        x, _ = _coord(1, c)
        parts.append(
            f'<text x="{x}" y="{MARGIN - 15}" text-anchor="middle" '
            f'font-size="14" font-family="monospace" fill="{LINE_COLOR}">'
            f'{COL_LABELS[c - 1]}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{BOARD_PX - 10}" text-anchor="middle" '
            f'font-size="14" font-family="monospace" fill="{LINE_COLOR}">'
            f'{COL_LABELS[c - 1]}</text>'
        )

    # Row labels (left and right)
    for r in range(1, BOARD_SIZE + 1):
        _, y = _coord(r, 1)
        parts.append(
            f'<text x="{MARGIN - 20}" y="{y + 5}" text-anchor="middle" '
            f'font-size="14" font-family="monospace" fill="{LINE_COLOR}">'
            f'{r}</text>'
        )
        parts.append(
            f'<text x="{BOARD_PX - MARGIN + 20}" y="{y + 5}" text-anchor="middle" '
            f'font-size="14" font-family="monospace" fill="{LINE_COLOR}">'
            f'{r}</text>'
        )

    # Stones
    last_point: Optional[Point] = None
    if game_state.moves:
        last_point = game_state.moves[-1].point

    for r in range(1, BOARD_SIZE + 1):
        for c in range(1, BOARD_SIZE + 1):
            pt = Point(r, c)
            player = game_state.board.get(pt)
            if player is None:
                continue
            x, y = _coord(r, c)
            fill = BLACK_STONE if player is Player.BLACK else WHITE_STONE
            stroke = "none" if player is Player.BLACK else WHITE_STROKE
            parts.append(
                f'<circle cx="{x}" cy="{y}" r="{STONE_RADIUS}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
            # Last move marker
            if highlight_last and pt == last_point:
                marker_color = WHITE_STONE if player is Player.BLACK else BLACK_STONE
                parts.append(
                    f'<circle cx="{x}" cy="{y}" r="6" '
                    f'fill="{marker_color}" opacity="0.7"/>'
                )

    # Clickable intersection targets — use opacity 0 + pointer-events:all
    # so they actually receive clicks (fill="transparent" does not in SVG)
    if clickable and not game_state.is_over:
        for r in range(1, BOARD_SIZE + 1):
            for c in range(1, BOARD_SIZE + 1):
                pt = Point(r, c)
                if not game_state.board.is_empty(pt):
                    continue
                x, y = _coord(r, c)
                col_label = COL_LABELS[c - 1]
                coord_str = f"{col_label}{r}"
                parts.append(
                    f'<circle cx="{x}" cy="{y}" r="{CLICK_RADIUS}" '
                    f'fill="black" opacity="0" pointer-events="all" '
                    f'class="board-click" data-coord="{coord_str}" '
                    f'style="cursor:pointer">'
                    f'<title>{coord_str}</title></circle>'
                )

    # Game-over overlay banner on the board itself
    if game_over_message:
        mid_y = BOARD_PX // 2
        # Semi-transparent backdrop
        parts.append(
            f'<rect x="0" y="{mid_y - 35}" width="{BOARD_PX}" height="70" '
            f'fill="black" opacity="0.65" rx="6"/>'
        )
        # Text color: green for win, red for loss, white for draw
        if "You win" in game_over_message:
            text_fill = "#4ADE80"
        elif "AI wins" in game_over_message:
            text_fill = "#F87171"
        else:
            text_fill = "#FFFFFF"
        parts.append(
            f'<text x="{BOARD_PX // 2}" y="{mid_y + 8}" '
            f'text-anchor="middle" font-size="26" font-weight="bold" '
            f'font-family="sans-serif" fill="{text_fill}">'
            f'{game_over_message}</text>'
        )

    parts.append("</svg>")

    # Inline JS: on every render, bind click listeners on the new SVG circles.
    # Uses onclick attributes would be simpler but Gradio may sanitize them,
    # so we use a <script> that runs immediately after the SVG is inserted.
    parts.append("""
<script>
(function() {
    var wrap = document.getElementById('gomoku-board-wrap');
    if (!wrap) return;
    wrap.addEventListener('click', function(e) {
        var circle = e.target.closest('.board-click');
        if (!circle) return;
        var coord = circle.getAttribute('data-coord');
        if (!coord) return;
        // Find the coord input — try textarea first (Gradio uses textarea for Textbox)
        var el = document.querySelector('#coord-input textarea')
              || document.querySelector('#coord-input input');
        if (!el) return;
        // Use native setter so Gradio's reactive system picks up the change
        var setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')
                  || Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
        if (setter && setter.set) {
            setter.set.call(el, coord);
        } else {
            el.value = coord;
        }
        el.dispatchEvent(new Event('input', {bubbles: true}));
        // Click the submit button (Gradio wraps it in a div with elem_id)
        var btn = document.querySelector('#coord-submit button')
               || document.querySelector('#coord-submit');
        if (btn) btn.click();
    });
})();
</script>""")

    parts.append("</div>")
    return "\n".join(parts)
