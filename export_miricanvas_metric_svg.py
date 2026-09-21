"""Export compact, outlined SVG metric charts for MiriCanvas/PowerPoint.

The output intentionally uses no SVG strokes and no live text. Chart lines are
expanded to filled polygons, and every visible character references an outlined
glyph path. Four editable colors are defined once in the SVG style block.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import re

import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath, TextToPath
import numpy as np
import pandas as pd


WIDTH = 1000.0
HEIGHT = 650.0
COLORS = {
    "k": "#000000",
    "w": "#FFFFFF",
    "b": "#2563EB",
    "r": "#D92D20",
}


def _number(value: float) -> str:
    value = 0.0 if abs(value) < 0.005 else value
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text or "0"


def _base36(number: int) -> str:
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    if number == 0:
        return "0"
    result = ""
    while number:
        number, remainder = divmod(number, 36)
        result = chars[remainder] + result
    return result


def _path_d(path: MplPath) -> str:
    parts: list[str] = []
    for vertices, code in path.iter_segments(simplify=False, curves=True):
        if code == MplPath.MOVETO:
            parts.append(f"M{_number(vertices[0])} {_number(vertices[1])}")
        elif code == MplPath.LINETO:
            parts.append(f"L{_number(vertices[0])} {_number(vertices[1])}")
        elif code == MplPath.CURVE3:
            parts.append(
                f"Q{_number(vertices[0])} {_number(vertices[1])} "
                f"{_number(vertices[2])} {_number(vertices[3])}"
            )
        elif code == MplPath.CURVE4:
            parts.append(
                f"C{_number(vertices[0])} {_number(vertices[1])} "
                f"{_number(vertices[2])} {_number(vertices[3])} "
                f"{_number(vertices[4])} {_number(vertices[5])}"
            )
        elif code == MplPath.CLOSEPOLY:
            parts.append("Z")
    return "".join(parts)


class OutlineText:
    def __init__(self) -> None:
        candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]
        available = {font.name for font in fm.fontManager.ttflist}
        family = next((name for name in candidates if name in available), "DejaVu Sans")
        self.prop = FontProperties(family=family, size=1)
        self.to_path = TextToPath()
        self.glyph_ids: dict[str, str] = {}
        self.glyph_paths: dict[str, str] = {}
        self.advance: dict[str, float] = {}
        self.string_ids: dict[str, str] = {}
        self.string_widths: dict[str, float] = {}

    def register(self, text: str) -> str:
        if text in self.string_ids:
            return self.string_ids[text]
        string_id = f"s{_base36(len(self.string_ids))}"
        self.string_ids[text] = string_id
        x = 0.0
        for character in text:
            if character not in self.advance:
                width, _, _ = self.to_path.get_text_width_height_descent(
                    character, self.prop, ismath=False
                )
                self.advance[character] = float(width)
                if not character.isspace():
                    glyph_id = f"g{_base36(len(self.glyph_ids))}"
                    self.glyph_ids[character] = glyph_id
                    self.glyph_paths[character] = _path_d(
                        TextPath((0, 0), character, prop=self.prop, size=1)
                    )
            x += self.advance[character]
        self.string_widths[text] = x
        return string_id

    def defs(self) -> str:
        glyphs = "".join(
            f'<path id="{self.glyph_ids[ch]}" d="{self.glyph_paths[ch]}"/>'
            for ch in self.glyph_ids
        )
        strings: list[str] = []
        for text, string_id in self.string_ids.items():
            x = 0.0
            uses: list[str] = []
            for character in text:
                glyph_id = self.glyph_ids.get(character)
                if glyph_id:
                    uses.append(f'<use href="#{glyph_id}" x="{_number(x)}"/>')
                x += self.advance[character]
            strings.append(f'<g id="{string_id}">{"".join(uses)}</g>')
        return glyphs + "".join(strings)

    def use(
        self,
        text: str,
        x: float,
        y: float,
        size: float,
        *,
        color_class: str = "k",
        anchor: str = "start",
        rotate: float = 0.0,
        opacity: float | None = None,
    ) -> str:
        string_id = self.register(text)
        width = self.string_widths[text] * size
        if anchor == "middle":
            x -= width / 2.0
        elif anchor == "end":
            x -= width
        transform = f"translate({_number(x)} {_number(y)})"
        if rotate:
            transform += f"rotate({_number(rotate)})"
        transform += f"scale({_number(size)} {_number(-size)})"
        opacity_attr = "" if opacity is None else f' opacity="{_number(opacity)}"'
        return f'<use href="#{string_id}" class="{color_class}" transform="{transform}"{opacity_attr}/>'


def _rect(x: float, y: float, width: float, height: float, cls: str, opacity: float | None = None) -> str:
    op = "" if opacity is None else f' opacity="{_number(opacity)}"'
    return (
        f'<rect class="{cls}" x="{_number(x)}" y="{_number(y)}" '
        f'width="{_number(width)}" height="{_number(height)}"{op}/>'
    )


def _circle(x: float, y: float, radius: float, cls: str) -> str:
    return f'<circle class="{cls}" cx="{_number(x)}" cy="{_number(y)}" r="{_number(radius)}"/>'


def _segment(x1: float, y1: float, x2: float, y2: float, width: float, cls: str) -> str:
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0:
        return _circle(x1, y1, width / 2.0, cls)
    ox, oy = -dy * width / (2.0 * length), dx * width / (2.0 * length)
    points = [
        (x1 + ox, y1 + oy), (x2 + ox, y2 + oy),
        (x2 - ox, y2 - oy), (x1 - ox, y1 - oy),
    ]
    d = "M" + "L".join(f"{_number(x)} {_number(y)}" for x, y in points) + "Z"
    return f'<path class="{cls}" d="{d}"/>'


def _polyline_shape(points: list[tuple[float, float]], width: float, cls: str) -> str:
    """Approximate an expanded polyline as one compact filled ribbon."""
    if len(points) < 2:
        return ""
    half = width / 2.0
    normals: list[tuple[float, float]] = []
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy)) or 1.0
        normals.append((-dy / length, dx / length))
    offsets: list[tuple[float, float]] = []
    for index in range(len(points)):
        if index == 0:
            nx, ny = normals[0]
        elif index == len(points) - 1:
            nx, ny = normals[-1]
        else:
            nx = normals[index - 1][0] + normals[index][0]
            ny = normals[index - 1][1] + normals[index][1]
            length = float(np.hypot(nx, ny)) or 1.0
            nx, ny = nx / length, ny / length
        offsets.append((nx * half, ny * half))
    left = [(x + ox, y + oy) for (x, y), (ox, oy) in zip(points, offsets)]
    right = [(x - ox, y - oy) for (x, y), (ox, oy) in zip(points, offsets)][::-1]
    polygon = left + right
    d = "M" + "L".join(f"{_number(x)} {_number(y)}" for x, y in polygon) + "Z"
    return f'<path class="{cls}" d="{d}"/>'


def _metric_limits(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if column == "r2":
        return -2.0, 1.05
    low, high = float(values.min()), float(values.max())
    padding = max(0.1, (high - low) * 0.08, max(abs(low), abs(high), 1.0) * 0.03)
    lower, upper = low - padding, high + padding
    if column in {"rmse", "mape"}:
        lower = 0.0
    elif column == "coverage":
        lower = max(0.0, lower)
        upper = min(100.5, max(upper, high + 0.5))
    return lower, upper


def _ticks(lower: float, upper: float, count: int = 5) -> np.ndarray:
    return np.linspace(lower, upper, count)


def _tick_label(value: float, column: str) -> str:
    if column in {"rmse", "mape", "coverage"} and abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _render_panel(
    text: OutlineText,
    frame: pd.DataFrame,
    column: str,
    title: str,
    cls: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    label_group_id: str,
) -> str:
    parts: list[str] = []
    lower, upper = _metric_limits(frame, column)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    values = np.clip(values, lower, upper)
    count = len(values)
    xs = np.linspace(x0 + 15.0, x0 + width - 8.0, count)

    def sy(value: float) -> float:
        return y0 + height - (value - lower) / (upper - lower) * height

    # Neutral grid and frame are filled rectangles, not strokes.
    for tick in _ticks(lower, upper):
        y = sy(float(tick))
        parts.append(_rect(x0, y - 0.35, width, 0.7, "k", 0.09))
        parts.append(text.use(_tick_label(float(tick), column), x0 - 5, y + 2.5, 7.0, anchor="end", opacity=0.72))
    parts.extend([
        _rect(x0, y0, width, 0.8, "k", 0.72),
        _rect(x0, y0 + height - 0.8, width, 0.8, "k", 0.72),
        _rect(x0, y0, 0.8, height, "k", 0.72),
        _rect(x0 + width - 0.8, y0, 0.8, height, "k", 0.72),
    ])

    if column == "coverage" and lower <= 98.0 <= upper:
        target_y = sy(98.0)
        dash = 7.0
        current = x0
        while current < x0 + width:
            parts.append(_rect(current, target_y - 0.6, min(dash, x0 + width - current), 1.2, "k", 0.75))
            current += dash * 1.8

    points = [(float(x), sy(float(v))) for x, v in zip(xs, values) if np.isfinite(v)]
    parts.append(_polyline_shape(points, 1.45, cls))
    # Keep alternating markers; the expanded line retains every data point,
    # while fewer circles keep each outlined SVG below the 100 KB limit.
    for index, (x, y) in enumerate(points):
        if index % 2 == 0 or index == len(points) - 1:
            parts.append(_circle(x, y, 1.75, cls))

    parts.append(text.use(title, x0 + width / 2, y0 - 8, 10.0, anchor="middle"))
    parts.append(
        f'<use href="#{label_group_id}" transform="translate({_number(x0)} {_number(y0)})"/>'
    )
    return "".join(parts)


def export_one(equipment_id: str, input_csv: Path, output_svg: Path) -> None:
    frame = pd.read_csv(input_csv)
    frame = frame.loc[frame["equipment_id"].astype(str).eq(equipment_id)].copy()
    frame["variable"] = frame["variable"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    frame["label"] = frame.get("label", frame["variable"]).astype(str).str.replace("\n", " ", regex=False).str.strip()
    frame = frame.sort_values("r2", ascending=False, na_position="last").drop_duplicates("variable")
    labels = frame["label"].fillna(frame["variable"]).tolist()

    text = OutlineText()
    main_title = f"{equipment_id} · Full 파인튜닝 · Tukey IQR 미적용 · 변수별 예측 성능"
    panels = [
        ("r2", "R² (높을수록 좋음)", "b", 42.0, 43.0),
        ("rmse", "RMSE (낮을수록 좋음)", "r", 525.0, 43.0),
        ("mape", "MAPE (%) (낮을수록 좋음)", "r", 42.0, 365.0),
        ("coverage", "Coverage (%) (목표 98%)", "b", 525.0, 365.0),
    ]

    # Register all text before emitting defs.
    text.register(main_title)
    for _, title, _, _, _ in panels:
        text.register(title)
    for label in labels:
        text.register(label)
    for column, _, _, _, _ in panels:
        low, high = _metric_limits(frame, column)
        for value in _ticks(low, high):
            text.register(_tick_label(float(value), column))

    # Variable labels are identical on all four panels. Define the outlined
    # label axis once and reuse it to reduce the SVG payload substantially.
    label_group_id = "la"
    label_xs = np.linspace(15.0, 433.0 - 8.0, len(labels))
    label_size = 4.25 if len(labels) >= 37 else 4.45
    label_group = '<g id="la">' + "".join(
        text.use(label, float(x) - 1.3, 174.0 + 6.0, label_size, rotate=90, opacity=0.82)
        for x, label in zip(label_xs, labels)
    ) + "</g>"

    body = [_rect(0, 0, WIDTH, HEIGHT, "w")]
    body.append(text.use(main_title, WIDTH / 2, 20, 14.0, anchor="middle"))
    for column, title, cls, x0, y0 in panels:
        body.append(
            _render_panel(text, frame, column, title, cls, x0, y0, 433.0, 174.0, label_group_id)
        )

    style = "".join(f".{key}{{fill:{value}}}" for key, value in COLORS.items())
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(WIDTH)}" height="{int(HEIGHT)}" '
        f'viewBox="0 0 {int(WIDTH)} {int(HEIGHT)}">'
        f'<style>{style}</style><defs>{text.defs()}{label_group}</defs>{"".join(body)}</svg>'
    )
    # Remove inter-tag whitespace defensively; the rest is already minified.
    svg = re.sub(r">\s+<", "><", svg)
    output_svg.write_text(svg, encoding="utf-8")


def main() -> None:
    output_dir = Path("model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    for equipment_id in ("2CM", "3CM", "4CM"):
        export_one(
            equipment_id,
            Path(f"chronos_metrics_{equipment_id}__review_v2_full_finetuned_test.csv"),
            output_dir / f"process_metric_lines_{equipment_id}_miricanvas.svg",
        )


if __name__ == "__main__":
    main()
