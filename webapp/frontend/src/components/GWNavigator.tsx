interface Props {
  currentIndex: number
  totalGWs: number
  currentGW: number
  onPrev: () => void
  onNext: () => void
  onSlide: (index: number) => void
}

export default function GWNavigator({ currentIndex, totalGWs, currentGW, onPrev, onNext, onSlide }: Props) {
  return (
    <div className="flex items-center gap-4">
      <button
        onClick={onPrev}
        disabled={currentIndex <= 0}
        className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
      >
        Prev
      </button>
      <div className="flex-1 flex items-center gap-3">
        <span className="text-lg font-bold whitespace-nowrap">GW {currentGW}</span>
        <input
          type="range"
          min={0}
          max={totalGWs - 1}
          value={currentIndex}
          onChange={e => onSlide(Number(e.target.value))}
          className="flex-1 accent-emerald-500"
        />
      </div>
      <button
        onClick={onNext}
        disabled={currentIndex >= totalGWs - 1}
        className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
      >
        Next
      </button>
    </div>
  )
}
