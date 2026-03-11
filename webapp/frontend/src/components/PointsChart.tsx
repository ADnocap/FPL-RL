import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts'
import { GameweekFrame } from '../types'

interface Props {
  gameweeks: GameweekFrame[]
  currentGWIndex: number
}

export default function PointsChart({ gameweeks, currentGWIndex }: Props) {
  const data = gameweeks.map(gw => ({
    gw: `GW${gw.gw}`,
    total: gw.total_points,
    net: gw.net_points,
  }))

  const currentPoint = data[currentGWIndex]

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 mb-3">Season Progress</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <XAxis dataKey="gw" tick={{ fontSize: 10, fill: '#9ca3af' }} interval={4} />
          <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px', fontSize: '12px' }}
            labelStyle={{ color: '#9ca3af' }}
          />
          <Line type="monotone" dataKey="total" stroke="#10b981" strokeWidth={2} dot={false} name="Total Points" />
          {currentPoint && (
            <ReferenceDot x={currentPoint.gw} y={currentPoint.total} r={5} fill="#10b981" stroke="#fff" strokeWidth={2} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
