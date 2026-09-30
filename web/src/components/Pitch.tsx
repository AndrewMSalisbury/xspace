import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { FramePayload, LayerName, Player } from '../types'

const L = 105
const W = 68

// Pitch coordinates are centred at (0, 0) with +y up; SVG has +y down.
const sx = (x: number) => x + L / 2
const sy = (y: number) => W / 2 - y

const SCALES: Record<LayerName, (t: number) => string> = {
  control: d3.interpolateRdBu,
  reach: d3.interpolateViridis,
  value: d3.interpolateMagma,
  xspace: d3.interpolateInferno,
}

function Heatmap({ frame, layer }: { frame: FramePayload; layer: LayerName }) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const { values } = frame.layers[layer]
    const rows = values.length
    const cols = values[0].length
    canvas.width = cols
    canvas.height = rows
    const ctx = canvas.getContext('2d')!
    const img = ctx.createImageData(cols, rows)
    // control uses a diverging scale reversed so attack = red
    const scale = layer === 'control' ? (t: number) => SCALES.control(1 - t) : SCALES[layer]
    for (let r = 0; r < rows; r++) {
      const src = values[rows - 1 - r] // flip: canvas row 0 is the top touchline (+y)
      for (let c = 0; c < cols; c++) {
        const col = d3.rgb(scale(src[c] / 255))
        const i = (r * cols + c) * 4
        img.data[i] = col.r
        img.data[i + 1] = col.g
        img.data[i + 2] = col.b
        img.data[i + 3] = 235
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [frame, layer])

  return <canvas ref={ref} className="heatmap" />
}

function PitchLines() {
  return (
    <g className="pitch-lines">
      <rect x={0} y={0} width={L} height={W} />
      <line x1={L / 2} y1={0} x2={L / 2} y2={W} />
      <circle cx={L / 2} cy={W / 2} r={9.15} />
      <rect x={0} y={(W - 40.32) / 2} width={16.5} height={40.32} />
      <rect x={L - 16.5} y={(W - 40.32) / 2} width={16.5} height={40.32} />
      <rect x={0} y={(W - 18.32) / 2} width={5.5} height={18.32} />
      <rect x={L - 5.5} y={(W - 18.32) / 2} width={5.5} height={18.32} />
    </g>
  )
}

function Players({ players, team }: { players: Player[]; team: 'att' | 'def' }) {
  return (
    <g className={`players ${team}`}>
      {players.map((p) => (
        <g key={p.id} transform={`translate(${sx(p.x)},${sy(p.y)})`}>
          <line x1={0} y1={0} x2={p.vx * 0.7} y2={-p.vy * 0.7} className="velocity" />
          <circle r={1.1} />
          <text dy={0.45}>{p.number}</text>
        </g>
      ))}
    </g>
  )
}

export function Pitch({ frame, layer }: { frame: FramePayload; layer: LayerName }) {
  const { shape } = frame
  return (
    <div className="pitch">
      <Heatmap frame={frame} layer={layer} />
      <svg viewBox={`-2 -2 ${L + 4} ${W + 4}`} preserveAspectRatio="xMidYMid meet">
        <PitchLines />
        <g className="def-lines">
          <line x1={sx(shape.mid_line)} x2={sx(shape.mid_line)} y1={0} y2={W} className="mid" />
          <line x1={sx(shape.back_line)} x2={sx(shape.back_line)} y1={0} y2={W} className="back" />
          <line x1={sx(shape.offside_line)} x2={sx(shape.offside_line)} y1={0} y2={W} className="offside" />
        </g>
        <Players players={frame.defenders} team="def" />
        <Players players={frame.attackers} team="att" />
        <circle cx={sx(frame.ball.x)} cy={sy(frame.ball.y)} r={0.7} className="ball" />
      </svg>
    </div>
  )
}
