import { PlayerInfo } from '../types'

const positionColors: Record<string, string> = {
  GK: 'bg-amber-500',
  DEF: 'bg-blue-500',
  MID: 'bg-green-500',
  FWD: 'bg-red-500',
}

interface Props {
  player: PlayerInfo
  small?: boolean
}

export default function PlayerCard({ player, small }: Props) {
  const size = small ? 'w-10 h-10 text-xs' : 'w-12 h-12 text-sm'
  const nameSize = small ? 'text-[10px]' : 'text-xs'

  return (
    <div className="flex flex-col items-center gap-0.5">
      {(player.is_captain || player.is_vice_captain) && (
        <span className="text-[10px] font-bold bg-yellow-400 text-black rounded-full w-4 h-4 flex items-center justify-center">
          {player.is_captain ? 'C' : 'V'}
        </span>
      )}
      <div className={`${size} ${positionColors[player.position]} rounded-full flex items-center justify-center font-bold text-white shadow-lg`}>
        {player.points}
      </div>
      <span className={`${nameSize} text-gray-300 text-center max-w-[80px] truncate`}>
        {player.name}
      </span>
    </div>
  )
}
