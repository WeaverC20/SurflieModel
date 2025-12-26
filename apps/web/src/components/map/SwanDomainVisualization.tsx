'use client'

import { useState, useEffect, useRef } from 'react'

interface SwanDomainVisualizationProps {
  apiBaseUrl?: string
  domainName?: string
}

interface BoundaryPoint {
  index: number
  lat: number
  lon: number
  depth_m: number
}

interface SwanDomainData {
  name: string
  region: string
  resolution_m: number
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
  n_lat: number
  n_lon: number
  n_wet_cells: number
  offshore_boundary_km: number
  bathymetry?: {
    lat: number[]
    lon: number[]
    depth: (number | null)[][]
    downsample_factor: number
  }
  boundary_points?: BoundaryPoint[]
  latest_run?: {
    run_id: string
    created_at: string
    swan_completed: boolean
    n_boundary_points: number
  }
}

export default function SwanDomainVisualization({
  apiBaseUrl = 'http://localhost:8000',
  domainName = 'california_swan_2000m'
}: SwanDomainVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [domainData, setDomainData] = useState<SwanDomainData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch domain data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)

        // Fetch with downsample=2 for faster loading of large domains
        const response = await fetch(
          `${apiBaseUrl}/api/swan/domain/${domainName}?include_bathymetry=true&include_boundary=true&downsample=2`
        )

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data: SwanDomainData = await response.json()
        console.log('Fetched SWAN domain:', {
          name: data.name,
          resolution: data.resolution_m,
          wet_cells: data.n_wet_cells,
          boundary_points: data.boundary_points?.length || 0,
        })

        setDomainData(data)
        setLoading(false)
      } catch (err) {
        console.error('Error fetching SWAN domain:', err)
        setError(err instanceof Error ? err.message : 'Unknown error')
        setLoading(false)
      }
    }

    fetchData()
  }, [apiBaseUrl, domainName])

  // Render visualization on canvas
  useEffect(() => {
    if (!domainData?.bathymetry || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { lat, lon, depth } = domainData.bathymetry
    const height = depth.length
    const width = depth[0].length

    // Calculate canvas size to fill container while maintaining aspect ratio
    const aspectRatio = width / height
    const containerWidth = canvas.parentElement?.clientWidth || 800

    canvas.width = Math.min(containerWidth, 1200)
    canvas.height = canvas.width / aspectRatio

    const cellWidth = canvas.width / width
    const cellHeight = canvas.height / height

    // Find max depth for normalization
    let maxDepth = 0
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const d = depth[i][j]
        if (d !== null && d > maxDepth) {
          maxDepth = d
        }
      }
    }

    console.log(`SWAN domain: ${width}x${height} cells, max depth: ${maxDepth.toFixed(0)}m`)

    // Draw bathymetry
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const d = depth[i][j]

        if (d === null) {
          // Land or outside domain - dark gray
          ctx.fillStyle = 'rgb(40, 40, 45)'
        } else {
          // Ocean - color by depth (deeper = darker blue)
          const normalized = Math.min(d / maxDepth, 1)
          ctx.fillStyle = depthToColor(normalized)
        }

        // Flip Y-axis
        ctx.fillRect(
          j * cellWidth,
          (height - 1 - i) * cellHeight,
          cellWidth + 0.5, // Slight overlap to avoid gaps
          cellHeight + 0.5
        )
      }
    }

    // Draw boundary points as black circles
    if (domainData.boundary_points && domainData.boundary_points.length > 0) {
      const latMin = lat[0]
      const latMax = lat[lat.length - 1]
      const lonMin = lon[0]
      const lonMax = lon[lon.length - 1]

      ctx.fillStyle = 'black'
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 2

      for (const point of domainData.boundary_points) {
        // Convert lat/lon to canvas coordinates
        const x = ((point.lon - lonMin) / (lonMax - lonMin)) * canvas.width
        const y = canvas.height - ((point.lat - latMin) / (latMax - latMin)) * canvas.height

        // Draw filled circle with white outline
        ctx.beginPath()
        ctx.arc(x, y, 6, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
      }
    }

    console.log('SWAN domain visualization rendered')
  }, [domainData])

  // Convert normalized depth (0-1) to color
  const depthToColor = (normalized: number): string => {
    // Light blue (shallow) -> Deep blue (deep)
    const r = Math.round(20 + (1 - normalized) * 80)
    const g = Math.round(60 + (1 - normalized) * 140)
    const b = Math.round(120 + (1 - normalized) * 135)
    return `rgb(${r}, ${g}, ${b})`
  }

  if (loading) {
    return (
      <div className="flex h-full min-h-[600px] items-center justify-center bg-slate-900">
        <div className="text-center">
          <div className="text-white text-xl mb-2">Loading SWAN Domain...</div>
          <div className="text-slate-400 text-sm">Fetching bathymetry and boundary data</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full min-h-[600px] items-center justify-center bg-slate-900">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-2">Error Loading SWAN Domain</div>
          <div className="text-slate-400 text-sm">{error}</div>
          <div className="text-slate-500 text-xs mt-4">
            Make sure the FastAPI server is running and SWAN data exists
          </div>
        </div>
      </div>
    )
  }

  if (!domainData) {
    return (
      <div className="flex h-full min-h-[600px] items-center justify-center bg-slate-900">
        <div className="text-white">No SWAN domain data available</div>
      </div>
    )
  }

  return (
    <div className="w-full bg-slate-900 p-4">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-white mb-2">
          SWAN Outer Domain - {domainData.region}
        </h2>
        <p className="text-slate-400 text-sm">
          Nearshore wave model domain with WW3 boundary conditions
        </p>
      </div>

      {/* Canvas Container */}
      <div className="bg-slate-800 rounded-lg p-4 mb-6">
        <canvas
          ref={canvasRef}
          className="w-full rounded border border-slate-700"
          style={{
            maxWidth: '100%',
            height: 'auto',
          }}
        />
      </div>

      {/* Info Panel */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {/* Domain Info */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-white font-semibold mb-3">Domain Configuration</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Resolution:</span>
              <span className="text-white">{domainData.resolution_m}m</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Grid Size:</span>
              <span className="text-white">{domainData.n_lat} × {domainData.n_lon}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Wet Cells:</span>
              <span className="text-white">{domainData.n_wet_cells?.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Offshore Boundary:</span>
              <span className="text-white">{domainData.offshore_boundary_km}km</span>
            </div>
          </div>
        </div>

        {/* Geographic Bounds */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-white font-semibold mb-3">Geographic Extent</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Latitude:</span>
              <span className="text-white">{domainData.lat_min?.toFixed(2)}° to {domainData.lat_max?.toFixed(2)}°N</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Longitude:</span>
              <span className="text-white">{domainData.lon_min?.toFixed(2)}° to {domainData.lon_max?.toFixed(2)}°W</span>
            </div>
          </div>
        </div>

        {/* Boundary Conditions */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-white font-semibold mb-3">WW3 Boundary</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Boundary Points:</span>
              <span className="text-white">{domainData.boundary_points?.length || 0}</span>
            </div>
            {domainData.latest_run && (
              <>
                <div className="flex justify-between">
                  <span className="text-slate-400">Latest Run:</span>
                  <span className="text-white">{domainData.latest_run.run_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Status:</span>
                  <span className={domainData.latest_run.swan_completed ? 'text-green-400' : 'text-yellow-400'}>
                    {domainData.latest_run.swan_completed ? 'Completed' : 'Prepared'}
                  </span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-white font-semibold mb-3">Legend</h3>
        <div className="flex flex-wrap gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded" style={{ background: 'linear-gradient(to bottom, rgb(100, 200, 255), rgb(20, 60, 120))' }} />
            <span className="text-slate-300">Bathymetry (shallow → deep)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-slate-700" />
            <span className="text-slate-300">Land / Outside Domain</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-full bg-black border-2 border-white" />
            <span className="text-slate-300">WW3 Boundary Condition Points</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="text-xs text-slate-500 mt-4 text-center">
        SWAN (Simulating WAves Nearshore) domain using GEBCO bathymetry with WW3 boundary conditions
      </div>
    </div>
  )
}
