'use client'

import { useState, useEffect, useRef } from 'react'

interface BuoyHeatmapProps {
  apiBaseUrl?: string
}

interface BuoyObservation {
  station_id: string
  name: string
  lat: number
  lon: number
  observation: any
  status: string
  error?: string
}

interface BuoyData {
  buoys: BuoyObservation[]
  count: number
  successful: number
  failed: number
  bounds: {
    min_lat: number
    max_lat: number
    min_lon: number
    max_lon: number
  }
}

export default function BuoyHeatmap({
  apiBaseUrl = 'http://localhost:8000'
}: BuoyHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [buoyData, setBuoyData] = useState<BuoyData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch buoy data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)

        const response = await fetch(`${apiBaseUrl}/api/buoys/california`)

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data: BuoyData = await response.json()
        console.log('Fetched buoy data:', {
          count: data.count,
          successful: data.successful,
          failed: data.failed,
          bounds: data.bounds
        })

        setBuoyData(data)
        setLoading(false)
      } catch (err) {
        console.error('Error fetching buoy data:', err)
        setError(err instanceof Error ? err.message : 'Unknown error')
        setLoading(false)
      }
    }

    fetchData()
  }, [apiBaseUrl])

  // Render buoy points on canvas
  useEffect(() => {
    if (!buoyData || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size to match ocean currents heatmap
    canvas.width = 400
    canvas.height = 628

    const { bounds } = buoyData

    // Calculate lat/lon to pixel conversion
    const latRange = bounds.max_lat - bounds.min_lat
    const lonRange = bounds.max_lon - bounds.min_lon

    const latToY = (lat: number) => {
      return canvas.height - ((lat - bounds.min_lat) / latRange) * canvas.height
    }

    const lonToX = (lon: number) => {
      return ((lon - bounds.min_lon) / lonRange) * canvas.width
    }

    // Clear canvas
    ctx.fillStyle = 'rgba(15, 23, 42, 1)' // slate-900
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Find max wave height for normalization
    let maxWaveHeight = 0
    for (const buoy of buoyData.buoys) {
      if (buoy.status === 'success' && buoy.observation?.wave_height_m !== null) {
        const height = buoy.observation.wave_height_m
        if (height > maxWaveHeight) {
          maxWaveHeight = height
        }
      }
    }

    console.log(`Max wave height: ${maxWaveHeight} m (${(maxWaveHeight * 3.28084).toFixed(1)} ft)`)

    // Draw buoy points
    for (const buoy of buoyData.buoys) {
      const x = lonToX(buoy.lon)
      const y = latToY(buoy.lat)

      if (buoy.status === 'success' && buoy.observation?.wave_height_m !== null) {
        const waveHeight = buoy.observation.wave_height_m
        const normalized = waveHeight / maxWaveHeight
        const color = waveHeightToColor(normalized)

        // Draw circle
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(x, y, 8, 0, 2 * Math.PI)
        ctx.fill()

        // Draw border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)'
        ctx.lineWidth = 2
        ctx.stroke()

        // Draw station label
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
        ctx.font = '10px monospace'
        ctx.fillText(buoy.station_id, x + 12, y + 4)
      } else {
        // Draw gray circle for failed buoys
        ctx.fillStyle = 'rgba(150, 150, 150, 0.5)'
        ctx.beginPath()
        ctx.arc(x, y, 6, 0, 2 * Math.PI)
        ctx.fill()
      }
    }

    console.log(`Rendered ${buoyData.successful} buoys with data, ${buoyData.failed} failed`)
  }, [buoyData])

  // Convert normalized wave height (0-1) to color
  const waveHeightToColor = (normalized: number): string => {
    // Blue (small waves) -> Cyan -> Green -> Yellow -> Red (large waves)
    if (normalized < 0.2) {
      // Blue to cyan
      const t = normalized / 0.2
      return `rgb(${Math.round(t * 100)}, ${Math.round(150 + t * 100)}, 255)`
    } else if (normalized < 0.4) {
      // Cyan to green
      const t = (normalized - 0.2) / 0.2
      return `rgb(${Math.round(100 + t * 155)}, 255, ${Math.round(255 - t * 155)})`
    } else if (normalized < 0.6) {
      // Green to yellow
      const t = (normalized - 0.4) / 0.2
      return `rgb(255, 255, ${Math.round(255 - t * 255)})`
    } else if (normalized < 0.8) {
      // Yellow to orange
      const t = (normalized - 0.6) / 0.2
      return `rgb(255, ${Math.round(255 - t * 105)}, 0)`
    } else {
      // Orange to red
      const t = (normalized - 0.8) / 0.2
      return `rgb(255, ${Math.round(150 - t * 150)}, 0)`
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-900">
        <div className="text-center">
          <div className="text-white text-xl mb-2">Loading Buoy Data...</div>
          <div className="text-slate-400 text-sm">Fetching NDBC observations</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-900">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-2">Error Loading Data</div>
          <div className="text-slate-400 text-sm">{error}</div>
          <div className="text-slate-500 text-xs mt-4">
            Make sure the FastAPI server is running on {apiBaseUrl}
          </div>
        </div>
      </div>
    )
  }

  if (!buoyData) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-900">
        <div className="text-white">No data available</div>
      </div>
    )
  }

  return (
    <div className="relative w-full min-h-full bg-slate-900 flex flex-col items-center p-8">
      {/* Canvas */}
      <div className="bg-slate-800 p-4 rounded-lg shadow-xl">
        <canvas
          ref={canvasRef}
          className="border border-slate-700 rounded"
          style={{
            imageRendering: 'auto',
            maxWidth: '100%',
            height: 'auto'
          }}
        />
      </div>

      {/* Info Panel */}
      <div className="mt-6 bg-slate-800/90 backdrop-blur-sm rounded-lg p-4 shadow-lg w-full max-w-2xl mb-8">
        <h3 className="text-white text-lg font-semibold mb-3">
          Real-Time Buoy Observations - California
        </h3>

        <div className="grid grid-cols-2 gap-4 text-sm text-slate-300 mb-4">
          <div>
            <span className="font-semibold">Latitude:</span> {buoyData.bounds.min_lat.toFixed(2)}°N to {buoyData.bounds.max_lat.toFixed(2)}°N
          </div>
          <div>
            <span className="font-semibold">Longitude:</span> {buoyData.bounds.min_lon.toFixed(2)}°E to {buoyData.bounds.max_lon.toFixed(2)}°E
          </div>
          <div>
            <span className="font-semibold">Total Buoys:</span> {buoyData.count}
          </div>
          <div>
            <span className="font-semibold">Reporting:</span> {buoyData.successful} / {buoyData.count}
          </div>
        </div>

        {/* Color Legend */}
        <div className="border-t border-slate-700 pt-3">
          <div className="font-semibold text-white text-sm mb-2">Significant Wave Height</div>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <div className="flex-1 h-6 rounded" style={{
              background: 'linear-gradient(to right, rgb(0,150,255), rgb(100,250,255), rgb(255,255,100), rgb(255,150,0), rgb(255,0,0))'
            }} />
            <div className="flex justify-between w-full text-xs">
              <span>0 ft</span>
              <span>Small</span>
              <span>Moderate</span>
              <span>Large</span>
              <span>Very Large</span>
            </div>
          </div>
          <div className="text-xs text-slate-500 mt-2">
            Gray circles indicate buoys with no current data
          </div>
        </div>

        {/* Buoy Details */}
        <div className="border-t border-slate-700 pt-3 mt-3 max-h-48 overflow-y-auto">
          <div className="font-semibold text-white text-sm mb-2">Buoy Details</div>
          <div className="space-y-1 text-xs">
            {buoyData.buoys
              .filter(b => b.status === 'success' && b.observation?.wave_height_m !== null)
              .sort((a, b) => (b.observation?.wave_height_m || 0) - (a.observation?.wave_height_m || 0))
              .slice(0, 10)
              .map(buoy => (
                <div key={buoy.station_id} className="flex justify-between text-slate-300">
                  <span>{buoy.name} ({buoy.station_id})</span>
                  <span className="font-mono">
                    {buoy.observation.wave_height_ft?.toFixed(1)} ft
                    ({buoy.observation.wave_height_m?.toFixed(2)} m)
                  </span>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}
