export interface PlayerInfo {
  element_id: number
  name: string
  position: 'GK' | 'DEF' | 'MID' | 'FWD'
  team: string
  points: number
  is_captain: boolean
  is_vice_captain: boolean
  purchase_price: number
  selling_price: number
}

export interface TransferInfo {
  element_id: number
  name: string
  position: string
  team: string
  price: number
}

export interface AutoSub {
  out_name: string
  in_name: string
}

export interface GameweekFrame {
  gw: number
  lineup: PlayerInfo[]
  bench: PlayerInfo[]
  formation: string
  transfers_in: TransferInfo[]
  transfers_out: TransferInfo[]
  chip_used: string | null
  gw_points: number
  hit_cost: number
  net_points: number
  total_points: number
  bank: number
  free_transfers: number
  auto_subs: AutoSub[]
  captain_failover: boolean
  chips_available: Record<string, boolean[]>
}

export interface SimulationResponse {
  season: string
  gameweeks: GameweekFrame[]
}
