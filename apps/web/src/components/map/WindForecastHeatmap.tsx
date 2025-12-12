'use client'

import { useState, useEffect, useRef } from 'react'
import ForecastHourSelector from '../ForecastHourSelector'

interface WindForecastHeatmapProps {
  apiBaseUrl?: string
}

interface WindGridData {
  lat: number[]
  lon: number[]
  u_wind: number[][]
  v_wind: number[][]
  wind_speed: number[][]
  wind_direction: number[][]
  forecast_time: string
  cycle_time: string
  forecast_hour: number
  resolution_deg: number
  model: string
  units: {
    wind_speed: string
    wind_direction: string
    lat: string
    lon: string
  }
}

export default function WindForecastHeatmap({
  apiBaseUrl = 'http://localhost:8000'
}: WindForecastHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [gridData, setGridData] = useState<WindGridData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [forecastHour, setForecastHour] = useState(0)

  // Fetch wind grid data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)

        const response = await fetch(`${apiBaseUrl}/api/wind/grid?forecast_hour=${forecastHour}`)

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data: WindGridData = await response.json()
        console.log('Fetched wind grid:', {
          model: data.model,
          forecast_time: data.forecast_time,
          resolution: data.resolution_deg,
          dimensions: `${data.lat.length} x ${data.lon.length}`
        })

        setGridData(data)
        setLoading(false)
      } catch (err) {
        console.error('Error fetching wind data:', err)
        setError(err instanceof Error ? err.message : 'Unknown error')
        setLoading(false)
      }
    }

    fetchData()
  }, [apiBaseUrl, forecastHour])

  // Render heatmap on canvas
  useEffect(() => {
    if (!gridData || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const height = gridData.wind_speed.length
    const width = gridData.wind_speed[0].length

    canvas.width = 400
    canvas.height = Math.round((height / width) * 400)

    const cellWidth = canvas.width / width
    const cellHeight = canvas.height / height

    // Find max wind speed for normalization
    let maxSpeed = 0
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const speed = gridData.wind_speed[i][j]
        if (speed !== null && !isNaN(speed) && speed > maxSpeed) {
          maxSpeed = speed
        }
      }
    }

    console.log(`Max wind speed: ${maxSpeed.toFixed(2)} m/s (${(maxSpeed * 1.94384).toFixed(1)} kts)`)

    // Draw each cell
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const speed = gridData.wind_speed[i][j]

        if (speed === null || isNaN(speed)) {
          // Land or missing data
          ctx.fillStyle = 'rgba(100, 100, 100, 0.2)'
        } else {
          // Wind - color by speed
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

    console.log('Wind heatmap rendered successfully')
  }, [gridData])

  // Convert normalized speed (0-1) to color
  const speedToColor = (normalized: number): string => {
    // Light blue (calm) -> Yellow -> Orange -> Red (strong winds)
    if (normalized < 0.2) {
      // Light blue to cyan
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
          <div className="text-white text-xl mb-2">Loading Wind Forecast...</div>
          <div className="text-slate-400 text-sm">Fetching GFS data</div>
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
      {/* Forecast Hour Selector */}
      <div className="mb-4 self-start">
        <ForecastHourSelector
          value={forecastHour}
          onChange={setForecastHour}
          maxHours={384}
          disabled={loading}
        />
      </div>

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
          Wind Forecast - GFS Model
        </h3>

        <div className="grid grid-cols-2 gap-4 text-sm text-slate-300 mb-4">
          <div>
            <span className="font-semibold">Model Run:</span>{' '}
            {new Date(gridData.cycle_time).toLocaleString('en-US', {
              timeZone: 'UTC',
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
              timeZoneName: 'short'
            })}
          </div>
          <div>
            <span className="font-semibold">Forecast Time:</span>{' '}
            {new Date(gridData.forecast_time).toLocaleString('en-US', {
              timeZone: 'UTC',
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
              timeZoneName: 'short'
            })}
          </div>
          <div>
            <span className="font-semibold">Resolution:</span> {gridData.resolution_deg}° (~25 km)
          </div>
          <div>
            <span className="font-semibold">Grid Size:</span> {gridData.lat.length} x {gridData.lon.length}
          </div>
        </div>

        {/* Color Legend */}
        <div className="border-t border-slate-700 pt-3">
          <div className="font-semibold text-white text-sm mb-2">Wind Speed (10m height)</div>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <div className="flex-1 h-6 rounded" style={{
              background: 'linear-gradient(to right, rgb(0,150,255), rgb(100,250,255), rgb(255,255,100), rgb(255,150,0), rgb(255,0,0))'
            }} />
            <div className="flex justify-between w-full text-xs">
              <span>0 kts</span>
              <span>Light</span>
              <span>Moderate</span>
              <span>Strong</span>
              <span>Very Strong</span>
            </div>
          </div>
        </div>

        <div className="text-xs text-slate-500 mt-3">
          Data from NOAA GFS (Global Forecast System)
        </div>
      </div>
    </div>
  )
}
