interface Props {
  name: string
  used: boolean
  active: boolean
}

const chipLabels: Record<string, string> = {
  wildcard: 'WC',
  free_hit: 'FH',
  bench_boost: 'BB',
  triple_captain: 'TC',
}

export default function ChipBadge({ name, used, active }: Props) {
  const label = chipLabels[name] || name
  let className = 'px-2 py-0.5 rounded-full text-xs font-bold '

  if (active) {
    className += 'bg-yellow-400 text-black animate-pulse'
  } else if (used) {
    className += 'bg-gray-700 text-gray-500 line-through'
  } else {
    className += 'bg-gray-700 text-gray-300'
  }

  return <span className={className}>{label}</span>
}
