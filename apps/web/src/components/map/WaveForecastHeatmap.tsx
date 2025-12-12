'use client'

import { useState, useEffect, useRef } from 'react'
import ForecastHourSelector from '../ForecastHourSelector'

interface WaveForecastHeatmapProps {
  apiBaseUrl?: string
}

interface WaveGridData {
  lat: number[]
  lon: number[]
  significant_wave_height: number[][]
  peak_wave_period: number[][]
  mean_wave_direction: number[][]
  wind_sea_height: number[][]
  swell_height: number[][]
  forecast_time: string
  cycle_time: string
  forecast_hour: number
  resolution_deg: number
  model: string
  units: {
    significant_wave_height: string
    peak_wave_period: string
    mean_wave_direction: string
    wind_sea_height: string
    swell_height: string
    lat: string
    lon: string
  }
}

export default function WaveForecastHeatmap({
  apiBaseUrl = 'http://localhost:8000'
}: WaveForecastHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [gridData, setGridData] = useState<WaveGridData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [forecastHour, setForecastHour] = useState(0)

  // Fetch wave grid data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)

        const response = await fetch(`${apiBaseUrl}/api/waves/grid?forecast_hour=${forecastHour}`)

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data: WaveGridData = await response.json()
        console.log('Fetched wave grid:', {
          model: data.model,
          forecast_time: data.forecast_time,
          resolution: data.resolution_deg,
          dimensions: `${data.lat.length} x ${data.lon.length}`
        })

        setGridData(data)
        setLoading(false)
      } catch (err) {
        console.error('Error fetching wave data:', err)
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
    const height = gridData.significant_wave_height.length
    const width = gridData.significant_wave_height[0].length

    canvas.width = 400
    canvas.height = Math.round((height / width) * 400)

    const cellWidth = canvas.width / width
    const cellHeight = canvas.height / height

    // Find max wave height for normalization
    let maxHeight = 0
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const waveHeight = gridData.significant_wave_height[i][j]
        if (waveHeight !== null && !isNaN(waveHeight) && waveHeight > maxHeight) {
          maxHeight = waveHeight
        }
      }
    }

    console.log(`Max wave height: ${maxHeight.toFixed(2)} m (${(maxHeight * 3.28084).toFixed(1)} ft)`)

    // Draw each cell
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const waveHeight = gridData.significant_wave_height[i][j]

        if (waveHeight === null || isNaN(waveHeight)) {
          // Land or missing data
          ctx.fillStyle = 'rgba(100, 100, 100, 0.2)'
        } else {
          // Waves - color by height
          const normalized = waveHeight / maxHeight
          const color = waveHeightToColor(normalized)
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

    console.log('Wave heatmap rendered successfully')
  }, [gridData])

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
          <div className="text-white text-xl mb-2">Loading Wave Forecast...</div>
          <div className="text-slate-400 text-sm">Fetching WaveWatch III data</div>
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
          Wave Forecast - WaveWatch III
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
        </div>

        <div className="text-xs text-slate-500 mt-3">
          Data from NOAA WaveWatch III Model
        </div>
      </div>
    </div>
  )
}
