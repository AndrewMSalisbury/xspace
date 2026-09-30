# Xspace web

React + TypeScript + D3 front end. It renders precomputed JSON exported by the Python package
(`src/xspace/export/web.py`); no computation happens in the browser.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # static site in dist/
```

`public/data/sample_frame.json` is one frame from the open IDSSE dataset (CC BY 4.0), regenerated
with `uv run python scripts/export_frame.py` from the repo root.
