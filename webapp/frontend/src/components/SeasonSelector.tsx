import { useState, useEffect } from 'react'
import { fetchSeasons } from '../api'

interface Props {
  onSimulate: (season: string) => void
  isLoading: boolean
}

export default function SeasonSelector({ onSimulate, isLoading }: Props) {
  const [seasons, setSeasons] = useState<string[]>([])
  const [selected, setSelected] = useState('')

  useEffect(() => {
    fetchSeasons().then(s => {
      setSeasons(s)
      if (s.length > 0) setSelected(s[s.length - 1])
    }).catch(() => setSeasons([]))
  }, [])

  return (
    <div className="flex items-center gap-3">
      <select
        value={selected}
        onChange={e => setSelected(e.target.value)}
        className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
      >
        {seasons.map(s => (
          <option key={s} value={s}>{s}</option>
        ))}
      </select>
      <button
        onClick={() => selected && onSimulate(selected)}
        disabled={isLoading || !selected}
        className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-semibold transition-colors"
      >
        {isLoading ? (
          <span className="flex items-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Simulating...
          </span>
        ) : 'Simulate'}
      </button>
    </div>
  )
}
