"""SVG board renderer + JavaScript click handler for Gradio."""

from __future__ import annotations

import math
from typing import Optional

from betagomoku.game.board import BOARD_SIZE, COL_LABELS, GomokuGameState
from betagomoku.game.types import Player, Point

# Layout constants
CELL_SIZE = 36
MARGIN = 40
BOARD_PX = MARGIN * 2 + CELL_SIZE * (BOARD_SIZE - 1)
STONE_RADIUS = 14
CLICK_RADIUS = 16  # Invisible click target radius
EVAL_BAR_WIDTH = 24

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


def _eval_to_pct(score: int) -> float:
    """Map eval score to 0.0-1.0 (fraction for black portion). Uses tanh for smooth clamping."""
    return 0.5 + 0.5 * math.tanh(score / 10_000)


def render_eval_bar(eval_score: int) -> str:
    """Return an SVG string for the evaluation bar."""
    pct = _eval_to_pct(eval_score)
    black_h = int(pct * BOARD_PX)
    white_h = BOARD_PX - black_h

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{EVAL_BAR_WIDTH}" height="{BOARD_PX}" '
        f'viewBox="0 0 {EVAL_BAR_WIDTH} {BOARD_PX}" '
        f'class="eval-bar">'
    )
    # White portion (top when black is winning less)
    parts.append(
        f'<rect x="0" y="0" width="{EVAL_BAR_WIDTH}" height="{white_h}" '
        f'fill="#F5F5F5"/>'
    )
    # Black portion (bottom grows as black gets stronger)
    parts.append(
        f'<rect x="0" y="{white_h}" width="{EVAL_BAR_WIDTH}" height="{black_h}" '
        f'fill="#1A1A1A"/>'
    )
    # Border
    parts.append(
        f'<rect x="0" y="0" width="{EVAL_BAR_WIDTH}" height="{BOARD_PX}" '
        f'fill="none" stroke="#888" stroke-width="1" rx="3"/>'
    )
    # Score label
    display_score = f"{eval_score / 1000:+.1f}k" if abs(eval_score) >= 1000 else f"{eval_score:+d}"
    label_y = BOARD_PX // 2
    # Choose text color based on which half the midpoint sits in
    text_color = "#F5F5F5" if pct > 0.5 else "#1A1A1A"
    parts.append(
        f'<text x="{EVAL_BAR_WIDTH // 2}" y="{label_y + 4}" '
        f'text-anchor="middle" font-size="10" font-weight="bold" '
        f'font-family="monospace" fill="{text_color}">'
        f'{display_score}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_board_svg(
    game_state: GomokuGameState,
    clickable: bool = True,
    highlight_last: bool = True,
    game_over_message: str = "",
    eval_score: Optional[int] = None,
) -> str:
    """Render the board as an SVG string wrapped in a div with click JS."""
    parts: list[str] = []

    # Wrapper div — use flex layout when eval bar is present
    if eval_score is not None:
        parts.append(
            f'<div id="gomoku-board-wrap" style="display:inline-flex;align-items:stretch;gap:6px;">'
        )
        parts.append(render_eval_bar(eval_score))
    else:
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

    # Star points (standard 15x15: center + 4 corners of 4-4 points)
    star_positions = [(4, 4), (4, 12), (12, 4), (12, 12), (8, 8)]
    for sr, sc in star_positions:
        sx, sy = _coord(sr, sc)
        parts.append(f'<circle cx="{sx}" cy="{sy}" r="3" fill="{LINE_COLOR}"/>')

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
    parts.append("</div>")
    return "\n".join(parts)


# JS to inject via demo.load(js=...). Sets up a document-level click
# delegator once. Gradio strips <script> from gr.HTML, so this is the
# only reliable way to run JS.
BOARD_CLICK_JS = """
() => {
    if (window._gomokuBound) return;
    window._gomokuBound = true;
    document.addEventListener('click', function(e) {
        var circle = e.target;
        // Walk up in case the click hit a child (e.g. <title>)
        while (circle && !circle.classList?.contains('board-click')) {
            circle = circle.parentElement;
        }
        if (!circle) return;
        var coord = circle.getAttribute('data-coord');
        if (!coord) return;

        // Find the Gradio textbox input element inside #coord-input
        var el = document.querySelector('#coord-input textarea')
              || document.querySelector('#coord-input input');
        if (!el) return;

        // Use the native value setter so Gradio detects the change
        var proto = el.tagName === 'TEXTAREA'
            ? HTMLTextAreaElement.prototype
            : HTMLInputElement.prototype;
        var setter = Object.getOwnPropertyDescriptor(proto, 'value');
        if (setter && setter.set) {
            setter.set.call(el, coord);
        } else {
            el.value = coord;
        }
        el.dispatchEvent(new Event('input', {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));

        // Small delay to let Gradio process the input, then click submit
        setTimeout(function() {
            var btn = document.querySelector('#coord-submit button')
                   || document.querySelector('#coord-submit');
            if (btn) btn.click();
        }, 50);
    });

    // --- Randomize "Processing" loading text ---
    if (!window._gomokuProgressBound) {
        window._gomokuProgressBound = true;
        var thinkingWords = [
            "Abstracting", "Accomplishing", "Actioning", "Actualizing",
            "Aligning", "Backpropagating", "Baking", "Bargaining",
            "Boomeranging", "Bootstrapping", "Brewing", "Caffeinating",
            "Calculating", "Calibrating", "Cerebrating", "Channeling",
            "Churning", "Clauding", "Coalescing", "Coaxing",
            "Cogitating", "Combobulating", "Compressing", "Computing",
            "Conjuring", "Considering", "Convincing", "Cooking",
            "Crafting", "Creating", "Cross-examining", "Crunching",
            "Daydreaming", "Decombobulating", "Deconvolving", "Decoupling",
            "Deliberating", "Denoising", "Determining", "Differentiating",
            "Doing", "Effecting", "Embedding", "Expanding",
            "Extrapolating", "Factorizing", "Fermenting", "Fiddling",
            "Finagling", "Forging", "Formalizing", "Forming",
            "Freestyling", "Generalizing", "Generating", "Hallucinating",
            "Harmonizing", "Hatching", "Herding", "Honking",
            "Hustling", "Hyperventilating", "Ideating", "Inferring",
            "Integrating", "Interpolating", "Interrogating", "Juggling",
            "Linearizing", "Looping", "Manifesting", "Mapping",
            "Marinating", "Massaging", "Mediating", "Metabolizing",
            "Moseying", "Mulling", "Mustering", "Musing",
            "Negotiating", "Noodling", "Normalizing", "Nudging",
            "Oscillating", "Percolating", "Persuading", "Placating",
            "Poking", "Pondering", "Ponderizing", "Postulating",
            "Processing", "Prodding", "Projecting", "Pruning",
            "Puttering", "Realigning", "Rebalancing", "Recontextualizing",
            "Rederiving", "Reframing", "Regularizing", "Reparameterizing",
            "Resonating", "Rethreading", "Reticulating", "Ricocheting",
            "Ruminating", "Sanity-checking", "Schlepping", "Shucking",
            "Simmering", "Smoothing", "Smooshing", "Spinning",
            "Spiraling", "Stabilizing", "Stewing", "Stress-testing",
            "Summoning", "Synthesizing", "Thinking", "Tinkering",
            "Translating", "Transmuting", "Triangulating", "Unfolding",
            "Unrolling", "Unspooling", "Untangling", "Untwisting",
            "Unwinding", "Vaporizing", "Vibing", "Vibrating",
            "Working", "Wrangling", "Wrestling"
        ];
        var pickedWord = null;
        var lastWord = null;
        function swapProcessing() {
            var found = false;
            var walker = document.createTreeWalker(
                document.body, NodeFilter.SHOW_TEXT
            );
            var node;
            while (node = walker.nextNode()) {
                var idx = node.data.indexOf('processing');
                if (idx === -1 && lastWord) {
                    idx = node.data.indexOf(lastWord);
                }
                if (idx !== -1) {
                    var oldWord = node.data.indexOf('processing') !== -1
                        ? 'processing' : lastWord;
                    if (!pickedWord) {
                        pickedWord = thinkingWords[
                            Math.floor(Math.random() * thinkingWords.length)
                        ];
                    }
                    node.replaceData(idx, oldWord.length, pickedWord);
                    lastWord = pickedWord;
                    found = true;
                }
            }
            if (!found) { pickedWord = null; lastWord = null; }
            requestAnimationFrame(swapProcessing);
        }
        requestAnimationFrame(swapProcessing);
    }
}
"""
