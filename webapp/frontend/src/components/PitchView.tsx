import { PlayerInfo } from '../types'
import PlayerCard from './PlayerCard'

interface Props {
  lineup: PlayerInfo[]
  formation: string
}

export default function PitchView({ lineup, formation }: Props) {
  // Group players by position
  const gk = lineup.filter(p => p.position === 'GK')
  const def_ = lineup.filter(p => p.position === 'DEF')
  const mid = lineup.filter(p => p.position === 'MID')
  const fwd = lineup.filter(p => p.position === 'FWD')

  return (
    <div className="relative w-full max-w-lg mx-auto aspect-[2/3] bg-gradient-to-b from-green-800 to-green-700 rounded-xl overflow-hidden shadow-2xl">
      {/* Pitch markings */}
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 200 300" fill="none" xmlns="http://www.w3.org/2000/svg">
        {/* Border */}
        <rect x="5" y="5" width="190" height="290" stroke="rgba(255,255,255,0.3)" strokeWidth="1" rx="4" />
        {/* Center line */}
        <line x1="5" y1="150" x2="195" y2="150" stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" />
        {/* Center circle */}
        <circle cx="100" cy="150" r="25" stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" />
        {/* Top penalty box */}
        <rect x="45" y="5" width="110" height="45" stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" />
        {/* Bottom penalty box */}
        <rect x="45" y="250" width="110" height="45" stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" />
        {/* Top 6-yard box */}
        <rect x="70" y="5" width="60" height="18" stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" />
        {/* Bottom 6-yard box */}
        <rect x="70" y="277" width="60" height="18" stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" />
      </svg>

      {/* Formation label */}
      <div className="absolute top-2 right-3 text-xs text-white/50 font-mono">{formation}</div>

      {/* Player rows */}
      <div className="absolute inset-0 flex flex-col justify-between py-4 px-2">
        {/* FWD row - top */}
        <Row players={fwd} />
        {/* MID row */}
        <Row players={mid} />
        {/* DEF row */}
        <Row players={def_} />
        {/* GK row - bottom */}
        <Row players={gk} />
      </div>
    </div>
  )
}

function Row({ players }: { players: PlayerInfo[] }) {
  return (
    <div className="flex justify-evenly items-center">
      {players.map(p => (
        <PlayerCard key={p.element_id} player={p} />
      ))}
    </div>
  )
}
