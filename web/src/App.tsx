import { useEffect, useState } from 'react'
import { Pitch } from './components/Pitch'
import type { FramePayload, LayerName } from './types'

const LAYERS: { key: LayerName; label: string; blurb: string }[] = [
  { key: 'xspace', label: 'Expected Space', blurb: 'control × reach × xT gained' },
  { key: 'control', label: 'Pitch control', blurb: 'who gets there first' },
  { key: 'reach', label: 'Reachability', blurb: 'can a pass get there' },
  { key: 'value', label: 'xT gained', blurb: 'is it worth going there' },
]

const ZONE_LABELS: Record<string, string> = {
  behind: 'Behind the line',
  between: 'Between the lines',
  wide: 'Wide of the block',
  in_front: 'In front',
}

function formatClock(period: number, seconds: number) {
  const total = seconds + (period === 2 ? 45 * 60 : 0)
  const m = Math.floor(total / 60)
  const s = Math.floor(total % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

export default function App() {
  const [frame, setFrame] = useState<FramePayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [layer, setLayer] = useState<LayerName>('xspace')

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/sample_frame.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setFrame)
      .catch((e: Error) => setError(e.message))
  }, [])

  const zoneMax = frame ? Math.max(...frame.zones.map((z) => frame.totals[z]), 1e-9) : 1

  return (
    <main>
      <header>
        <h1>
          <span className="brand">X</span>space
        </h1>
        <p className="tagline">Expected Space — the room a defence leaves, and who uses it.</p>
      </header>

      {error && <p className="error">Couldn't load frame data ({error}). Run scripts/export_frame.py first.</p>}

      {frame && (
        <section className="viewer">
          <div className="viewer-head">
            <div>
              <strong>{frame.attacking_team}</strong> attacking → vs {frame.defending_team}
            </div>
            <div className="clock">{formatClock(frame.period, frame.time_s)}</div>
          </div>

          <div className="layer-tabs" role="tablist">
            {LAYERS.map((l) => (
              <button
                key={l.key}
                role="tab"
                aria-selected={layer === l.key}
                className={layer === l.key ? 'active' : ''}
                onClick={() => setLayer(l.key)}
              >
                {l.label}
                <small>{l.blurb}</small>
              </button>
            ))}
          </div>

          <Pitch frame={frame} layer={layer} />

          <div className="legend">
            <span><i className="swatch att" /> Attacking</span>
            <span><i className="swatch def" /> Defending</span>
            <span><i className="dash mid" /> Midfield line</span>
            <span><i className="dash back" /> Back line</span>
            <span><i className="dash offside" /> Offside line</span>
          </div>

          <div className="zones">
            <h2>
              Expected Space by zone <small>xT·m²</small>
            </h2>
            {frame.zones.map((z) => (
              <div className="zone-row" key={z}>
                <span className="zone-name">{ZONE_LABELS[z] ?? z}</span>
                <span className="bar">
                  <span style={{ width: `${(frame.totals[z] / zoneMax) * 100}%` }} />
                </span>
                <span className="zone-val">{frame.totals[z].toFixed(2)}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <footer>
        Tracking data: IDSSE open dataset (Bassek et al., 2025, CC BY 4.0). xT grid: Karun Singh.
        Pitch control after Spearman (2018).
      </footer>
    </main>
  )
}
