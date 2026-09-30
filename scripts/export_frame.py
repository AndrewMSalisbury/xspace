"""Export one frame's Expected Space surfaces for the web viewer.

    uv run python scripts/export_frame.py --match J03WMX --frame 3000
"""

from __future__ import annotations

import argparse

from xspace.export.web import frame_payload, write_json
from xspace.io.loaders import IDSSE_MATCHES, load_idsse
from xspace.metrics.space import frame_space
from xspace.physics.pitch_control import make_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WMX", choices=IDSSE_MATCHES)
    ap.add_argument("--frame", type=int, default=3000)
    ap.add_argument("--cell", type=float, default=1.0)
    ap.add_argument("--out", default="web/public/data/sample_frame.json")
    args = ap.parse_args()

    match = load_idsse(args.match)
    xs, ys, grid = make_grid(args.cell)
    fs = frame_space(match, args.frame, grid)
    if fs is None:
        raise SystemExit("no possession information for that frame; try another")
    path = write_json(frame_payload(match, fs, xs, ys), args.out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
