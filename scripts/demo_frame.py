"""Render the Expected Space components for one frame (IDSSE or PFF).

    uv run python scripts/demo_frame.py --match J03WMX --frame 3000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

from xspace.io.loaders import load_match
from xspace.metrics.space import frame_space, orient
from xspace.physics.pitch_control import make_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="idsse", choices=["idsse", "pff"])
    ap.add_argument("--match", default="J03WMX",
                    help="IDSSE id (e.g. J03WMX) or PFF id (e.g. 3812)")
    ap.add_argument("--frame", type=int, default=3000)
    ap.add_argument("--cell", type=float, default=1.0)
    ap.add_argument("--out", default="docs/img/demo_frame.png")
    args = ap.parse_args()

    t0 = time.perf_counter()
    match = load_match(args.source, args.match)
    print(f"loaded {match.n_frames} frames in {time.perf_counter() - t0:.1f}s")

    xs, ys, grid = make_grid(args.cell)
    t0 = time.perf_counter()
    fs = frame_space(match, args.frame, grid)
    print(f"frame computed in {time.perf_counter() - t0:.2f}s")
    if fs is None:
        raise SystemExit("no possession information for that frame; try another")
    print({k: round(v, 4) for k, v in fs.totals.items()})

    side = fs.attacking_side
    att, dfn = (match.home, match.away) if side == 0 else (match.away, match.home)
    ap_, dp_ = (orient(match.team_arrays(s)[0][args.frame], side) for s in (side, 1 - side))
    ball = orient(match.ball[args.frame], side)

    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=3,
                  pitch_color="#101418", line_color="#9aa4ad")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax in axes:
        pitch.draw(ax=ax)
    panels = [
        (fs.control.attack, f"Pitch control — {att.name}", "RdBu_r", (0, 1)),
        (fs.reach, "Pass reachability", "viridis", (0, 1)),
        (fs.value, "xT gained vs. ball position", "magma", None),
        (fs.xspace, "Expected Space = control × reach × xT gained", "inferno", None),
    ]
    extent = (0, 105, 0, 68)
    for ax, (surf, title, cmap, lims) in zip(axes, panels, strict=True):
        img = surf.reshape(len(ys), len(xs))
        vmin, vmax = lims if lims else (0, np.nanmax(img))
        ax.imshow(img, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                  alpha=0.85, zorder=1)
        ax.scatter(ap_[:, 0] + 52.5, ap_[:, 1] + 34, c="#e63946", s=60, ec="white", zorder=4)
        ax.scatter(dp_[:, 0] + 52.5, dp_[:, 1] + 34, c="#4ea8de", s=60, ec="white", zorder=4)
        ax.scatter(ball[0] + 52.5, ball[1] + 34, c="white", s=30, ec="black", zorder=5)
        for line_x, style in ((fs.shape.offside_line, ":"), (fs.shape.back_line, "-"),
                              (fs.shape.mid_line, "--")):
            ax.axvline(line_x + 52.5, color="#f1faee", lw=1, ls=style, alpha=0.6, zorder=3)
        ax.set_title(title, color="white", fontsize=11)
    fig.patch.set_facecolor("#101418")
    fig.suptitle(f"{match.home.name} vs. {match.away.name} — frame {args.frame} — "
                 f"{att.name} attacking →",
                 color="white")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, facecolor=fig.get_facecolor())
    print(f"saved {args.out}  (red = {att.name}, blue = {dfn.name})")


if __name__ == "__main__":
    main()
