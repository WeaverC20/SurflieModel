'use client'

import { useState, useEffect, useRef, useCallback } from 'react'

interface SwanOutputVisualizationProps {
  apiBaseUrl?: string
  domainName?: string
}

interface SwellComponent {
  hs: number[][] | null
  tp: number[][] | null
  dir: number[][] | null
}

interface SwanOutputData {
  domain: string
  run_id: string
  created_at: string
  swan_completed: boolean
  status: string
  grid: {
    lat: number[]
    lon: number[]
    combined: { hsig: number[][], tpeak: number[][], dir: number[][] }
    windsea: SwellComponent
    swell1: SwellComponent
    swell2: SwellComponent
    swell3: SwellComponent
    swell4: SwellComponent
    swell5: SwellComponent
    swell6: SwellComponent
    downsample_factor: number
  }
}

interface TooltipData {
  x: number
  y: number
  lat: number
  lon: number
  combined: { hs: number, tp: number, dir: number } | null
  swells: Array<{ name: string, hs: number, tp: number, dir: number }>
}

export default function SwanOutputVisualization({
  apiBaseUrl = 'http://localhost:8000',
  domainName = 'california_swan_5000m'
}: SwanOutputVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [outputData, setOutputData] = useState<SwanOutputData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [tooltip, setTooltip] = useState<TooltipData | null>(null)

  // Fetch SWAN output data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)

        const response = await fetch(
          `${apiBaseUrl}/api/swan/output/${domainName}?downsample=2`
        )

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data: SwanOutputData = await response.json()
        console.log('Fetched SWAN output:', {
          domain: data.domain,
          run_id: data.run_id,
          status: data.status,
          grid_size: data.grid ? `${data.grid.lat?.length} x ${data.grid.lon?.length}` : 'N/A'
        })

        if (data.status !== 'complete' || !data.grid) {
          throw new Error(`SWAN output not available: ${data.status}`)
        }

        setOutputData(data)
        setLoading(false)
      } catch (err) {
        console.error('Error fetching SWAN output:', err)
        setError(err instanceof Error ? err.message : 'Unknown error')
        setLoading(false)
      }
    }

    fetchData()
  }, [apiBaseUrl, domainName])

  // Render heatmap on canvas
  useEffect(() => {
    if (!outputData?.grid || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { lat, lon, combined } = outputData.grid
    const hsig = combined.hsig
    if (!hsig || !lat || !lon) return

    const height = hsig.length
    const width = hsig[0].length

    // Calculate canvas size
    const aspectRatio = width / height
    const containerWidth = canvas.parentElement?.clientWidth || 800

    canvas.width = Math.min(containerWidth, 1200)
    canvas.height = canvas.width / aspectRatio

    const cellWidth = canvas.width / width
    const cellHeight = canvas.height / height

    // Find max wave height for color scaling
    let maxHs = 0
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const h = hsig[i][j]
        if (h !== null && !isNaN(h) && h > maxHs) {
          maxHs = h
        }
      }
    }

    console.log(`SWAN output: ${width}x${height} cells, max Hs: ${maxHs.toFixed(2)}m`)

    // Draw wave height heatmap
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const h = hsig[i][j]

        if (h === null || isNaN(h)) {
          // Land or outside domain - dark gray
          ctx.fillStyle = 'rgb(40, 40, 45)'
        } else {
          // Ocean - color by wave height
          const normalized = Math.min(h / Math.max(maxHs, 3), 1) // Cap at 3m for better color distribution
          ctx.fillStyle = waveHeightToColor(normalized)
        }

        // Flip Y-axis (lat 0 = bottom)
        ctx.fillRect(
          j * cellWidth,
          (height - 1 - i) * cellHeight,
          cellWidth + 0.5,
          cellHeight + 0.5
        )
      }
    }

    console.log('SWAN output visualization rendered')
  }, [outputData])

  // Handle mouse move for tooltip
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!outputData?.grid || !canvasRef.current) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const { lat, lon, combined, windsea, swell1, swell2, swell3, swell4, swell5, swell6 } = outputData.grid
    const hsig = combined.hsig
    if (!hsig || !lat || !lon) return

    const height = hsig.length
    const width = hsig[0].length

    // Convert canvas coords to grid indices
    const cellWidth = canvas.width / width
    const cellHeight = canvas.height / height

    const j = Math.floor(x / cellWidth)
    const i = height - 1 - Math.floor(y / cellHeight)

    if (i < 0 || i >= height || j < 0 || j >= width) {
      setTooltip(null)
      return
    }

    // Get combined values
    const hs = combined.hsig?.[i]?.[j]
    if (hs === null || isNaN(hs)) {
      setTooltip(null)
      return
    }

    const tp = combined.tpeak?.[i]?.[j] || 0
    const dir = combined.dir?.[i]?.[j] || 0

    // Collect all swell components
    const swells: Array<{ name: string, hs: number, tp: number, dir: number }> = []

    const swellSources = [
      { name: 'Wind Sea', data: windsea },
      { name: 'Swell 1', data: swell1 },
      { name: 'Swell 2', data: swell2 },
      { name: 'Swell 3', data: swell3 },
      { name: 'Swell 4', data: swell4 },
      { name: 'Swell 5', data: swell5 },
      { name: 'Swell 6', data: swell6 },
    ]

    for (const { name, data } of swellSources) {
      if (!data?.hs) continue
      const swellHs = data.hs[i]?.[j]
      if (swellHs && !isNaN(swellHs) && swellHs > 0.05) { // Only show if > 5cm
        swells.push({
          name,
          hs: swellHs,
          tp: data.tp?.[i]?.[j] || 0,
          dir: data.dir?.[i]?.[j] || 0
        })
      }
    }

    // Sort swells by height (descending)
    swells.sort((a, b) => b.hs - a.hs)

    setTooltip({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      lat: lat[i],
      lon: lon[j],
      combined: { hs, tp, dir },
      swells
    })
  }, [outputData])

  const handleMouseLeave = useCallback(() => {
    setTooltip(null)
  }, [])

  // Convert wave height (0-1 normalized) to color
  const waveHeightToColor = (normalized: number): string => {
    // Blue (small) -> Cyan -> Green -> Yellow -> Orange -> Red (large)
    if (normalized < 0.2) {
      // Blue to Cyan
      const t = normalized / 0.2
      return `rgb(${Math.round(30 + t * 30)}, ${Math.round(100 + t * 155)}, ${Math.round(200 + t * 55)})`
    } else if (normalized < 0.4) {
      // Cyan to Green
      const t = (normalized - 0.2) / 0.2
      return `rgb(${Math.round(60 - t * 30)}, ${Math.round(255 - t * 55)}, ${Math.round(255 - t * 155)})`
    } else if (normalized < 0.6) {
      // Green to Yellow
      const t = (normalized - 0.4) / 0.2
      return `rgb(${Math.round(30 + t * 225)}, ${Math.round(200)}, ${Math.round(100 - t * 100)})`
    } else if (normalized < 0.8) {
      // Yellow to Orange
      const t = (normalized - 0.6) / 0.2
      return `rgb(${Math.round(255)}, ${Math.round(200 - t * 80)}, ${Math.round(0)})`
    } else {
      // Orange to Red
      const t = (normalized - 0.8) / 0.2
      return `rgb(${Math.round(255 - t * 55)}, ${Math.round(120 - t * 80)}, ${Math.round(0)})`
    }
  }

  // Convert direction in degrees to compass
  const dirToCompass = (deg: number): string => {
    const dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    const idx = Math.round(deg / 22.5) % 16
    return dirs[idx]
  }

  // Convert meters to feet
  const mToFt = (m: number): string => (m * 3.28084).toFixed(1)

  if (loading) {
    return (
      <div className="flex h-full min-h-[600px] items-center justify-center bg-slate-900">
        <div className="text-center">
          <div className="text-white text-xl mb-2">Loading SWAN Output...</div>
          <div className="text-slate-400 text-sm">Fetching wave model results</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full min-h-[600px] items-center justify-center bg-slate-900">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-2">Error Loading SWAN Output</div>
          <div className="text-slate-400 text-sm">{error}</div>
          <div className="text-slate-500 text-xs mt-4">
            Make sure SWAN has been run and output exists
          </div>
        </div>
      </div>
    )
  }

  if (!outputData) {
    return (
      <div className="flex h-full min-h-[600px] items-center justify-center bg-slate-900">
        <div className="text-white">No SWAN output data available</div>
      </div>
    )
  }

  return (
    <div className="w-full bg-slate-900 p-4" ref={containerRef}>
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-white mb-2">
          SWAN Wave Output - {outputData.domain}
        </h2>
        <p className="text-slate-400 text-sm">
          Nearshore wave forecast from SWAN model • Run: {outputData.run_id}
        </p>
      </div>

      {/* Canvas Container with Tooltip */}
      <div className="relative bg-slate-800 rounded-lg p-4 mb-6">
        <canvas
          ref={canvasRef}
          className="w-full rounded border border-slate-700 cursor-crosshair"
          style={{ maxWidth: '100%', height: 'auto' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />

        {/* Tooltip */}
        {tooltip && (
          <div
            className="absolute z-10 bg-slate-800/95 border border-slate-600 rounded-lg p-3 shadow-xl pointer-events-none"
            style={{
              left: Math.min(tooltip.x + 15, (containerRef.current?.clientWidth || 800) - 250),
              top: tooltip.y + 15,
              minWidth: '200px'
            }}
          >
            <div className="text-xs text-slate-400 mb-2">
              {tooltip.lat.toFixed(3)}°N, {Math.abs(tooltip.lon).toFixed(3)}°W
            </div>

            {tooltip.combined && (
              <div className="border-b border-slate-600 pb-2 mb-2">
                <div className="text-white font-semibold">
                  Combined: {mToFt(tooltip.combined.hs)}ft @ {tooltip.combined.tp.toFixed(1)}s {dirToCompass(tooltip.combined.dir)}
                </div>
                <div className="text-slate-400 text-xs">
                  ({tooltip.combined.hs.toFixed(2)}m, {tooltip.combined.dir.toFixed(0)}°)
                </div>
              </div>
            )}

            {tooltip.swells.length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-slate-500 uppercase">Components:</div>
                {tooltip.swells.map((swell, idx) => (
                  <div key={idx} className="text-sm text-slate-300">
                    <span className="text-slate-400">{swell.name}:</span>{' '}
                    {mToFt(swell.hs)}ft @ {swell.tp.toFixed(1)}s {dirToCompass(swell.dir)}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Color Legend */}
      <div className="bg-slate-800 rounded-lg p-4 mb-4">
        <h3 className="text-white font-semibold mb-3">Wave Height Legend</h3>
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-sm">0ft</span>
          <div
            className="flex-1 h-4 rounded"
            style={{
              background: 'linear-gradient(to right, rgb(30, 100, 200), rgb(60, 255, 255), rgb(30, 200, 100), rgb(255, 200, 0), rgb(255, 120, 0), rgb(200, 40, 0))'
            }}
          />
          <span className="text-slate-400 text-sm">10ft+</span>
        </div>
        <div className="mt-2 flex items-center gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded" style={{ background: 'rgb(40, 40, 45)' }} />
            <span>Land/Outside</span>
          </div>
        </div>
      </div>

      {/* Run Info */}
      <div className="text-xs text-slate-500 text-center">
        SWAN output from run {outputData.run_id} • Created: {new Date(outputData.created_at).toLocaleString()}
      </div>
    </div>
  )
}
