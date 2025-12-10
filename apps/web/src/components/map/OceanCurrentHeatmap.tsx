'use client'

import { useState, useEffect, useRef } from 'react'

interface OceanCurrentHeatmapProps {
  apiBaseUrl?: string
  region?: string
  forecastHour?: number
}

interface GridData {
  lats: number[]
  lons: number[]
  current_speed: (number | null)[][]
  current_direction: (number | null)[][]
  bounds: {
    min_lat: number
    max_lat: number
    min_lon: number
    max_lon: number
  }
  shape: {
    height: number
    width: number
  }
  metadata: any
}

export default function OceanCurrentHeatmap({
  apiBaseUrl = 'http://localhost:8000',
  region = 'california',
  forecastHour = 0
}: OceanCurrentHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [gridData, setGridData] = useState<GridData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch ocean current data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)

        const response = await fetch(
          `${apiBaseUrl}/api/ocean-currents/grid?region=${region}&forecast_hour=${forecastHour}`
        )

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data: GridData = await response.json()
        console.log('Fetched ocean current data:', {
          bounds: data.bounds,
          shape: data.shape,
          metadata: data.metadata
        })

        setGridData(data)
        setLoading(false)
      } catch (err) {
        console.error('Error fetching ocean current data:', err)
        setError(err instanceof Error ? err.message : 'Unknown error')
        setLoading(false)
      }
    }

    fetchData()
  }, [apiBaseUrl, region, forecastHour])

  // Render heatmap on canvas
  useEffect(() => {
    if (!gridData || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { current_speed, shape } = gridData
    const { width, height } = shape

    // Set canvas size
    canvas.width = width * 4 // Scale up for visibility
    canvas.height = height * 4

    console.log(`Rendering heatmap: ${width}x${height} -> ${canvas.width}x${canvas.height}`)

    // Find max speed for normalization (excluding null values)
    let maxSpeed = 0
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const speed = current_speed[i][j]
        if (speed !== null && speed > maxSpeed) {
          maxSpeed = speed
        }
      }
    }

    console.log(`Max current speed: ${maxSpeed} m/s (${(maxSpeed * 1.94384).toFixed(2)} knots)`)

    // Draw heatmap
    const cellWidth = canvas.width / width
    const cellHeight = canvas.height / height

    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const speed = current_speed[i][j]

        if (speed === null) {
          // Land - draw transparent/gray
          ctx.fillStyle = 'rgba(100, 100, 100, 0.2)'
        } else {
          // Ocean - color by speed
          const normalizedSpeed = speed / maxSpeed
          const color = speedToColor(normalizedSpeed)
          ctx.fillStyle = color
        }

        // Flip Y-axis: canvas Y goes top-to-bottom, but lat goes bottom-to-top
        ctx.fillRect(
          j * cellWidth,
          (height - 1 - i) * cellHeight,
          cellWidth,
          cellHeight
        )
      }
    }

    console.log('Heatmap rendered successfully')
  }, [gridData])

  // Convert normalized speed (0-1) to color
  const speedToColor = (normalized: number): string => {
    // Blue (slow) -> Cyan -> Green -> Yellow -> Red (fast)
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
          <div className="text-white text-xl mb-2">Loading Ocean Currents...</div>
          <div className="text-slate-400 text-sm">Fetching RTOFS data</div>
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

  if (!gridData) {
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
            imageRendering: 'pixelated',
            maxWidth: '100%',
            height: 'auto'
          }}
        />
      </div>

      {/* Info Panel */}
      <div className="mt-6 bg-slate-800/90 backdrop-blur-sm rounded-lg p-4 shadow-lg w-full max-w-2xl mb-8">
        <h3 className="text-white text-lg font-semibold mb-3">
          Ocean Current Heatmap - {region.toUpperCase()}
        </h3>

        <div className="grid grid-cols-2 gap-4 text-sm text-slate-300 mb-4">
          <div>
            <span className="font-semibold">Region:</span> {gridData.bounds.min_lat.toFixed(2)}°N to {gridData.bounds.max_lat.toFixed(2)}°N
          </div>
          <div>
            <span className="font-semibold">Longitude:</span> {gridData.bounds.min_lon.toFixed(2)}°E to {gridData.bounds.max_lon.toFixed(2)}°E
          </div>
          <div>
            <span className="font-semibold">Grid Size:</span> {gridData.shape.width} × {gridData.shape.height}
          </div>
          <div>
            <span className="font-semibold">Forecast Hour:</span> {forecastHour}
          </div>
        </div>

        {/* Color Legend */}
        <div className="border-t border-slate-700 pt-3">
          <div className="font-semibold text-white text-sm mb-2">Current Speed</div>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <div className="flex-1 h-6 rounded" style={{
              background: 'linear-gradient(to right, rgb(0,150,255), rgb(100,250,255), rgb(255,255,100), rgb(255,150,0), rgb(255,0,0))'
            }} />
            <div className="flex justify-between w-full text-xs">
              <span>0 knots</span>
              <span>Slow</span>
              <span>Moderate</span>
              <span>Fast</span>
              <span>Very Fast</span>
            </div>
          </div>
          <div className="text-xs text-slate-500 mt-2">
            Gray areas indicate land (no ocean current data)
          </div>
        </div>
      </div>
    </div>
  )
}
