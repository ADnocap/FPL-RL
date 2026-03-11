import { SimulationResponse } from './types'

const BASE_URL = '/api'

export async function fetchSeasons(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/seasons`)
  if (!res.ok) throw new Error('Failed to fetch seasons')
  const data = await res.json()
  return data.seasons
}

export async function simulateSeason(season: string): Promise<SimulationResponse> {
  const res = await fetch(`${BASE_URL}/simulate/${season}`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Simulation failed' }))
    throw new Error(err.detail || 'Simulation failed')
  }
  return res.json()
}
