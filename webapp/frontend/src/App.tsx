import { useSimulation } from './hooks/useSimulation'
import SeasonSelector from './components/SeasonSelector'
import GWNavigator from './components/GWNavigator'
import PitchView from './components/PitchView'
import BenchRow from './components/BenchRow'
import StatsBar from './components/StatsBar'
import TransferPanel from './components/TransferPanel'
import PointsChart from './components/PointsChart'

export default function App() {
  const { data, currentFrame, currentGWIndex, totalGWs, isLoading, error, simulate, setGW, nextGW, prevGW } = useSimulation()

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold">FPL Season Visualizer</h1>
          <SeasonSelector onSimulate={simulate} isLoading={isLoading} />
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6">
        {error && (
          <div className="bg-red-900/50 border border-red-700 rounded-lg px-4 py-3 mb-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {isLoading && (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <svg className="animate-spin h-10 w-10 mx-auto mb-4 text-emerald-500" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <p className="text-gray-400">Running simulation... This may take a moment.</p>
            </div>
          </div>
        )}

        {!isLoading && !data && !error && (
          <div className="flex items-center justify-center py-20">
            <p className="text-gray-500 text-lg">Select a season and click Simulate to get started.</p>
          </div>
        )}

        {currentFrame && data && (
          <div className="space-y-4">
            {/* GW Navigator */}
            <GWNavigator
              currentIndex={currentGWIndex}
              totalGWs={totalGWs}
              currentGW={currentFrame.gw}
              onPrev={prevGW}
              onNext={nextGW}
              onSlide={setGW}
            />

            {/* Stats Bar */}
            <StatsBar frame={currentFrame} />

            {/* Main content grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Pitch - takes 2 columns on large screens */}
              <div className="lg:col-span-2">
                <PitchView lineup={currentFrame.lineup} formation={currentFrame.formation} />
                <BenchRow bench={currentFrame.bench} />

                {/* Auto-subs */}
                {currentFrame.auto_subs.length > 0 && (
                  <div className="mt-2 max-w-lg mx-auto">
                    <div className="bg-gray-800/50 rounded-lg px-3 py-2">
                      <span className="text-xs text-gray-500">Auto-subs: </span>
                      {currentFrame.auto_subs.map((sub, i) => (
                        <span key={i} className="text-xs">
                          <span className="text-red-400">{sub.out_name}</span>
                          {' \u2192 '}
                          <span className="text-green-400">{sub.in_name}</span>
                          {i < currentFrame.auto_subs.length - 1 && ', '}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {currentFrame.captain_failover && (
                  <div className="mt-1 max-w-lg mx-auto">
                    <span className="text-xs text-yellow-400 bg-yellow-400/10 rounded px-2 py-1">
                      Captain failover to vice-captain
                    </span>
                  </div>
                )}
              </div>

              {/* Side panel */}
              <div className="space-y-4">
                <TransferPanel
                  transfers_in={currentFrame.transfers_in}
                  transfers_out={currentFrame.transfers_out}
                />
                <PointsChart gameweeks={data.gameweeks} currentGWIndex={currentGWIndex} />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
