import dynamic from 'next/dynamic'

// Dynamically import components with no SSR to avoid window/document issues
const WindForecastHeatmap = dynamic(() => import('@/components/map/WindForecastHeatmap'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-slate-900">
      <div className="text-center">
        <h1 className="text-2xl font-bold text-white mb-2">
          Loading Wind Forecast...
        </h1>
        <p className="text-slate-400 text-sm">
          Fetching GFS data
        </p>
      </div>
    </div>
  ),
})

const WaveForecastHeatmap = dynamic(() => import('@/components/map/WaveForecastHeatmap'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-slate-900">
      <div className="text-center">
        <h1 className="text-2xl font-bold text-white mb-2">
          Loading Wave Forecast...
        </h1>
        <p className="text-slate-400 text-sm">
          Fetching WaveWatch III data
        </p>
      </div>
    </div>
  ),
})

export default function ConditionsPage() {
  return (
    <main className="h-screen w-screen overflow-hidden bg-slate-900">
      <div className="flex h-full">
        {/* Left panel: Wind Forecast */}
        <div className="flex-1 border-r border-slate-700 overflow-y-auto">
          <WindForecastHeatmap />
        </div>

        {/* Right panel: Wave Forecast */}
        <div className="flex-1 overflow-y-auto">
          <WaveForecastHeatmap />
        </div>
      </div>
    </main>
  )
}
