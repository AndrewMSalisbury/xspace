// Mirrors the payload written by src/xspace/export/web.py

export type Player = {
  id: string
  number: number
  position: string
  x: number
  y: number
  vx: number
  vy: number
}

export type Layer = {
  max: number
  values: number[][] // [row = y index][col = x index], quantised 0-255
}

export type LayerName = 'control' | 'reach' | 'value' | 'xspace'

export type FramePayload = {
  match_id: string
  frame: number
  period: number
  time_s: number
  attacking_team: string
  defending_team: string
  ball: { x: number; y: number }
  attackers: Player[]
  defenders: Player[]
  grid: { xs: number[]; ys: number[] }
  layers: Record<LayerName, Layer>
  shape: { offside_line: number; back_line: number; mid_line: number; block_y: [number, number] }
  zones: string[]
  totals: Record<string, number>
}
