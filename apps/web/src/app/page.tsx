import dynamic from 'next/dynamic'

// Dynamically import components with no SSR to avoid window/document issues
const OceanCurrentHeatmap = dynamic(() => import('@/components/map/OceanCurrentHeatmap'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-slate-900">
      <div className="text-center">
        <h1 className="text-2xl font-bold text-white mb-2">
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
        <h1 className="text-2xl font-bold text-white mb-2">
          Loading Buoy Data...
        </h1>
        <p className="text-slate-400 text-sm">
          Fetching NDBC observations
        </p>
      </div>
    </div>
  ),
})

export default function Home() {
  return (
    <main className="h-screen w-screen overflow-hidden bg-slate-900">
      <div className="flex h-full">
        {/* Left panel: Ocean Currents */}
        <div className="flex-1 border-r border-slate-700 overflow-y-auto">
          <OceanCurrentHeatmap />
        </div>

        {/* Right panel: Buoy Observations */}
        <div className="flex-1 overflow-y-auto">
          <BuoyHeatmap />
        </div>
      </div>
    </main>
  )
}
