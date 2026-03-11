import { PlayerInfo } from '../types'
import PlayerCard from './PlayerCard'

interface Props {
  bench: PlayerInfo[]
}

export default function BenchRow({ bench }: Props) {
  return (
    <div className="flex justify-evenly items-center bg-gray-800/50 rounded-lg py-3 px-2 mt-2 max-w-lg mx-auto">
      <span className="text-xs text-gray-500 mr-2">BENCH</span>
      {bench.map((p, i) => (
        <PlayerCard key={p.element_id} player={p} small />
      ))}
    </div>
  )
}
