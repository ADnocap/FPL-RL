import { TransferInfo } from '../types'

interface Props {
  transfers_in: TransferInfo[]
  transfers_out: TransferInfo[]
}

export default function TransferPanel({ transfers_in, transfers_out }: Props) {
  if (transfers_in.length === 0 && transfers_out.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 mb-2">Transfers</h3>
        <p className="text-xs text-gray-500">No transfers this gameweek</p>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 mb-3">Transfers</h3>
      <div className="space-y-2">
        {transfers_in.map((tin, i) => (
          <div key={tin.element_id} className="flex items-center gap-2 text-sm">
            <div className="flex-1 flex items-center gap-2">
              {transfers_out[i] && (
                <>
                  <span className="text-red-400 font-medium">{transfers_out[i].name}</span>
                  <span className="text-gray-500 text-xs">{transfers_out[i].position}</span>
                  <span className="text-gray-600 text-xs">&pound;{(transfers_out[i].price / 10).toFixed(1)}m</span>
                </>
              )}
            </div>
            <span className="text-gray-500">&rarr;</span>
            <div className="flex-1 flex items-center gap-2">
              <span className="text-green-400 font-medium">{tin.name}</span>
              <span className="text-gray-500 text-xs">{tin.position}</span>
              <span className="text-gray-600 text-xs">&pound;{(tin.price / 10).toFixed(1)}m</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
