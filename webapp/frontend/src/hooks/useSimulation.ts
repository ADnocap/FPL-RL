import { useState, useCallback } from 'react'
import { SimulationResponse, GameweekFrame } from '../types'
import { simulateSeason } from '../api'

export function useSimulation() {
  const [data, setData] = useState<SimulationResponse | null>(null)
  const [currentGWIndex, setCurrentGWIndex] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const simulate = useCallback(async (season: string) => {
    setIsLoading(true)
    setError(null)
    setData(null)
    setCurrentGWIndex(0)
    try {
      const result = await simulateSeason(season)
      setData(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const currentFrame: GameweekFrame | null = data ? data.gameweeks[currentGWIndex] ?? null : null
  const totalGWs = data ? data.gameweeks.length : 0

  const setGW = useCallback((index: number) => {
    setCurrentGWIndex(Math.max(0, Math.min(index, totalGWs - 1)))
  }, [totalGWs])

  const nextGW = useCallback(() => {
    setCurrentGWIndex(prev => Math.min(prev + 1, totalGWs - 1))
  }, [totalGWs])

  const prevGW = useCallback(() => {
    setCurrentGWIndex(prev => Math.max(prev - 1, 0))
  }, [])

  return { data, currentFrame, currentGWIndex, totalGWs, isLoading, error, simulate, setGW, nextGW, prevGW }
}
