import dynamic from 'next/dynamic'

// Dynamically import components with no SSR to avoid window/document issues
const WindForecastHeatmap = dynamic(() => import('@/components/map/WindForecastHeatmap'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-slate-900">
      <div className="text-center">
        <h1 className="text-xl font-bold text-white mb-2">
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
        <h1 className="text-xl font-bold text-white mb-2">
          Loading Wave Forecast...
        </h1>
        <p className="text-slate-400 text-sm">
          Fetching WaveWatch III data
        </p>
      </div>
    </div>
  ),
})

const OceanCurrentHeatmap = dynamic(() => import('@/components/map/OceanCurrentHeatmap'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-slate-900">
      <div className="text-center">
        <h1 className="text-xl font-bold text-white mb-2">
          Loading Ocean Currents...
        </h1>
        <p className="text-slate-400 text-sm">
          Fetching RTOFS data
        </p>
      </div>
    </div>
  ),
})

const BuoyHeatmap = dynamic(() => import('@/components/map/BuoyHeatmap'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-slate-900">
      <div className="text-center">
        <h1 className="text-xl font-bold text-white mb-2">
          Loading Buoy Data...
        </h1>
        <p className="text-slate-400 text-sm">
          Fetching NDBC observations
        </p>
      </div>
    </div>
  ),
})

const SwanDomainVisualization = dynamic(() => import('@/components/map/SwanDomainVisualization'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full min-h-[600px] items-center justify-center bg-slate-900">
      <div className="text-center">
        <h1 className="text-xl font-bold text-white mb-2">
          Loading SWAN Domain...
        </h1>
        <p className="text-slate-400 text-sm">
          Fetching bathymetry and boundary data
        </p>
      </div>
    </div>
  ),
})

const SwanOutputVisualization = dynamic(() => import('@/components/map/SwanOutputVisualization'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full min-h-[600px] items-center justify-center bg-slate-900">
      <div className="text-center">
        <h1 className="text-xl font-bold text-white mb-2">
          Loading SWAN Output...
        </h1>
        <p className="text-slate-400 text-sm">
          Fetching wave model results
        </p>
      </div>
    </div>
  ),
})

export default function Home() {
  return (
    <main className="min-h-screen w-screen bg-slate-900">
      <div className="p-4">
        <h1 className="text-2xl font-bold text-white mb-4 text-center">
          California Ocean Conditions Dashboard
        </h1>

        {/* 2x2 Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Top Left: Wind Forecast */}
          <div className="border border-slate-700 rounded-lg overflow-hidden">
            <WindForecastHeatmap />
          </div>

          {/* Top Right: Wave Forecast */}
          <div className="border border-slate-700 rounded-lg overflow-hidden">
            <WaveForecastHeatmap />
          </div>

          {/* Bottom Left: Ocean Currents */}
          <div className="border border-slate-700 rounded-lg overflow-hidden">
            <OceanCurrentHeatmap />
          </div>

          {/* Bottom Right: Buoy Observations */}
          <div className="border border-slate-700 rounded-lg overflow-hidden">
            <BuoyHeatmap />
          </div>
        </div>

        {/* Full Width: SWAN Outer Domain Visualization */}
        <div className="mt-6 border border-slate-700 rounded-lg overflow-hidden">
          <SwanDomainVisualization domainName="california_swan_5000m" />
        </div>

        {/* Full Width: SWAN Wave Output Visualization */}
        <div className="mt-6 border border-slate-700 rounded-lg overflow-hidden">
          <SwanOutputVisualization domainName="california_swan_5000m" />
        </div>
      </div>
    </main>
  )
}
