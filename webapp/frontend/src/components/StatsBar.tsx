import { GameweekFrame } from '../types'
import ChipBadge from './ChipBadge'

interface Props {
  frame: GameweekFrame
}

export default function StatsBar({ frame }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-3 bg-gray-800 rounded-lg px-4 py-3">
      <div className="flex flex-col items-center">
        <span className="text-2xl font-bold text-white">{frame.net_points}</span>
        <span className="text-[10px] text-gray-400">GW PTS</span>
      </div>
      {frame.hit_cost > 0 && (
        <span className="text-xs text-red-400">(-{frame.hit_cost} hit)</span>
      )}
      <div className="w-px h-8 bg-gray-700" />
      <div className="flex flex-col items-center">
        <span className="text-2xl font-bold text-emerald-400">{frame.total_points}</span>
        <span className="text-[10px] text-gray-400">TOTAL</span>
      </div>
      <div className="w-px h-8 bg-gray-700" />
      <div className="flex flex-col items-center">
        <span className="text-sm font-medium text-white">&pound;{(frame.bank / 10).toFixed(1)}m</span>
        <span className="text-[10px] text-gray-400">BANK</span>
      </div>
      <div className="w-px h-8 bg-gray-700" />
      <div className="flex flex-col items-center">
        <span className="text-sm font-medium text-white">{frame.free_transfers}</span>
        <span className="text-[10px] text-gray-400">FTs</span>
      </div>
      <div className="w-px h-8 bg-gray-700" />
      <div className="flex items-center gap-1">
        {Object.entries(frame.chips_available).map(([chip, halves]) => {
          const anyAvailable = halves.some(h => h)
          const isActive = frame.chip_used === chip
          return (
            <ChipBadge key={chip} name={chip} used={!anyAvailable} active={isActive} />
          )
        })}
      </div>
    </div>
  )
}
