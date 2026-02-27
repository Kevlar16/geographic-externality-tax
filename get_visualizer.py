#!/usr/bin/env python3
"""Generate GIS-esque GET formula maps as standalone SVG files (no external deps).

Modes:
- generic: arbitrary 2D plane
- usa: contiguous U.S. approximate polygon in lon/lat space
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


@dataclass
class ModelConfig:
    base_rate: float = 0.05
    beta: float = 1.0
    rl_ratio: float = 1.0
    distance_reference: float = 1.0


def parse_hearths(text: str) -> List[Point]:
    points: List[Point] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_str, y_str = chunk.split(":")
        points.append((float(x_str), float(y_str)))
    if not points:
        raise ValueError("At least one hearth must be provided")
    return points


def parse_weights(text: str, n: int) -> List[float]:
    if not text.strip():
        return [1.0 / n] * n
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(values) != n:
        raise ValueError(f"Expected {n} weights, got {len(values)}")
    if any(v < 0 for v in values):
        raise ValueError("Weights must be non-negative")
    total = sum(values)
    if total <= 0:
        raise ValueError("Weights sum must be positive")
    return [v / total for v in values]


def euclidean(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def effective_distance(point: Point, hearths: Sequence[Point], weights: Sequence[float], metric: str) -> float:
    total = 0.0
    for (hx, hy), w in zip(hearths, weights):
        if metric == "euclidean":
            d = euclidean(point, (hx, hy))
        else:
            d = haversine_km(point[0], point[1], hx, hy)
        total += w * d
    return total


def tax_intensity(distance: float, cfg: ModelConfig) -> float:
    d_ref = cfg.distance_reference if cfg.distance_reference > 0 else 1.0
    return cfg.base_rate * (1 + cfg.beta * distance / d_ref) * cfg.rl_ratio


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def turbo_color(v: float) -> Tuple[int, int, int]:
    # lightweight blue->cyan->yellow->red gradient
    v = max(0.0, min(1.0, v))
    if v < 0.33:
        t = v / 0.33
        r, g, b = (0, int(255 * t), 255)
    elif v < 0.66:
        t = (v - 0.33) / 0.33
        r, g, b = (int(255 * t), 255, int(255 * (1 - t)))
    else:
        t = (v - 0.66) / 0.34
        r, g, b = (255, int(255 * (1 - t)), 0)
    return r, g, b


def to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def render_svg(
    output: str,
    bounds: Tuple[float, float, float, float],
    hearths: Sequence[Point],
    weights: Sequence[float],
    cfg: ModelConfig,
    resolution: int,
    title: str,
    mask_polygon: Sequence[Point] | None,
    metric: str,
) -> None:
    min_x, max_x, min_y, max_y = bounds
    dx = (max_x - min_x) / resolution
    dy = (max_y - min_y) / resolution

    d_vals: List[List[float | None]] = []
    d_max = 0.0

    for row in range(resolution):
        y = min_y + (row + 0.5) * dy
        row_vals: List[float | None] = []
        for col in range(resolution):
            x = min_x + (col + 0.5) * dx
            p = (x, y)
            if mask_polygon is not None and not point_in_polygon(p, mask_polygon):
                row_vals.append(None)
                continue
            d = effective_distance(p, hearths, weights, metric)
            row_vals.append(d)
            d_max = max(d_max, d)
        d_vals.append(row_vals)

    cfg2 = ModelConfig(cfg.base_rate, cfg.beta, cfg.rl_ratio, max(d_max, 1e-9))

    width = 1000
    height = 700
    margin = 40
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    rect_w = plot_w / resolution
    rect_h = plot_h / resolution

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#101820"/>',
        f'<text x="{width/2}" y="24" fill="white" text-anchor="middle" font-size="20">{title}</text>',
    ]

    for row in range(resolution):
        for col in range(resolution):
            d = d_vals[row][col]
            if d is None:
                continue
            t = tax_intensity(d, cfg2)
            # normalize based on possible range at d in [0, d_ref]
            t_min = cfg2.base_rate * cfg2.rl_ratio
            t_max = cfg2.base_rate * (1 + cfg2.beta) * cfg2.rl_ratio
            v = 0.0 if t_max == t_min else (t - t_min) / (t_max - t_min)
            color = to_hex(turbo_color(v))

            px = margin + col * rect_w
            py = margin + (resolution - 1 - row) * rect_h
            parts.append(
                f'<rect x="{px:.2f}" y="{py:.2f}" width="{rect_w + 0.2:.2f}" height="{rect_h + 0.2:.2f}" fill="{color}" stroke="none"/>'
            )

    # hearth markers
    for i, (hx, hy) in enumerate(hearths):
        px = margin + (hx - min_x) / (max_x - min_x) * plot_w
        py = margin + (max_y - hy) / (max_y - min_y) * plot_h
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" fill="#ffffff"/>')
        parts.append(
            f'<text x="{px + 8:.2f}" y="{py - 8:.2f}" fill="white" font-size="11">H{i+1} ({weights[i]:.2f})</text>'
        )

    # optional USA boundary
    if mask_polygon is not None:
        pts = []
        for x, y in mask_polygon:
            px = margin + (x - min_x) / (max_x - min_x) * plot_w
            py = margin + (max_y - y) / (max_y - min_y) * plot_h
            pts.append(f"{px:.2f},{py:.2f}")
        parts.append(f'<polygon points="{" ".join(pts)}" fill="none" stroke="#ffffff" stroke-width="1.2"/>')

    parts.append("</svg>")
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def usa_contiguous_polygon() -> List[Point]:
    # Simplified contiguous U.S. outline (lon, lat) for visualization only
    return [
        (-124.7, 48.5), (-124.0, 42.0), (-122.0, 40.0), (-121.0, 38.0), (-119.0, 36.0),
        (-117.0, 34.0), (-115.0, 33.0), (-111.0, 31.5), (-106.5, 31.5), (-103.0, 29.5),
        (-99.0, 27.0), (-94.0, 28.5), (-90.0, 29.0), (-86.0, 30.0), (-83.0, 29.0),
        (-81.0, 26.0), (-80.0, 28.0), (-80.0, 31.0), (-79.0, 33.0), (-77.0, 35.0),
        (-75.0, 38.0), (-74.0, 40.5), (-70.0, 43.0), (-69.5, 45.0), (-72.0, 46.0),
        (-77.0, 44.5), (-82.0, 45.0), (-88.0, 47.0), (-95.0, 49.0), (-110.0, 49.0),
        (-124.7, 48.5),
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create GIS-esque GET formula maps as SVG")
    p.add_argument("--mode", choices=["generic", "usa"], default="generic")
    p.add_argument("--hearths", required=True, help="Comma-separated x:y points")
    p.add_argument("--weights", default="", help="Comma-separated weights")
    p.add_argument("--base-rate", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--rl", type=float, default=1.0)
    p.add_argument("--resolution", type=int, default=120)
    p.add_argument("--output", default="get_map.svg")
    p.add_argument("--min-x", type=float, default=-10)
    p.add_argument("--max-x", type=float, default=10)
    p.add_argument("--min-y", type=float, default=-10)
    p.add_argument("--max-y", type=float, default=10)
    return p


def main() -> None:
    args = build_parser().parse_args()
    hearths = parse_hearths(args.hearths)
    weights = parse_weights(args.weights, len(hearths))
    cfg = ModelConfig(args.base_rate, args.beta, args.rl)

    if args.mode == "generic":
        render_svg(
            output=args.output,
            bounds=(args.min_x, args.max_x, args.min_y, args.max_y),
            hearths=hearths,
            weights=weights,
            cfg=cfg,
            resolution=args.resolution,
            title="GET Tax Intensity (Generic 2D)",
            mask_polygon=None,
            metric="euclidean",
        )
    else:
        usa_poly = usa_contiguous_polygon()
        min_x = min(x for x, _ in usa_poly)
        max_x = max(x for x, _ in usa_poly)
        min_y = min(y for _, y in usa_poly)
        max_y = max(y for _, y in usa_poly)
        render_svg(
            output=args.output,
            bounds=(min_x, max_x, min_y, max_y),
            hearths=hearths,
            weights=weights,
            cfg=cfg,
            resolution=args.resolution,
            title="GET Tax Intensity (USA Approximation)",
            mask_polygon=usa_poly,
            metric="haversine",
        )


if __name__ == "__main__":
    main()
