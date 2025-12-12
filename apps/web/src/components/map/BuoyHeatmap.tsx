'use client'

import { useState, useEffect, useRef } from 'react'

interface BuoyHeatmapProps {
  apiBaseUrl?: string
}

interface SwellComponent {
  height_m: number | null
  height_ft: number | null
  period_s: number | null
  direction_deg: number | null
  direction_cardinal: string | null
}

interface SpectralData {
  swell?: SwellComponent
  wind_waves?: SwellComponent
  combined?: {
    significant_height_m: number | null
    significant_height_ft: number | null
    average_period_s: number | null
    mean_direction_deg: number | null
    mean_direction_cardinal: string | null
    steepness: string | null
  }
  partitions?: Array<{
    partition_id: number
    height_m: number
    height_ft: number
    period_s: number | null
    direction_deg: number | null
    type: string
  }>
}

interface BuoyObservation {
  station_id: string
  name: string
  network: string
  lat: number
  lon: number
  depth_m?: number
  observation: any
  spectral_data?: SpectralData | null
  status: string
  error?: string
}

interface BuoyData {
  buoys: BuoyObservation[]
  count: number
  successful: number
  failed: number
  networks: {
    ndbc: number
    cdip: number
  }
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
  const [selectedBuoy, setSelectedBuoy] = useState<BuoyObservation | null>(null)
  const [includeSpectral, setIncludeSpectral] = useState(false)
  const [showNetworkFilter, setShowNetworkFilter] = useState({ ndbc: true, cdip: true })

  // Fetch buoy data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)

        const params = new URLSearchParams({
          include_ndbc: showNetworkFilter.ndbc.toString(),
          include_cdip: showNetworkFilter.cdip.toString(),
          include_spectral: includeSpectral.toString(),
        })

        const response = await fetch(`${apiBaseUrl}/api/buoys/california?${params}`)

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data: BuoyData = await response.json()
        console.log('Fetched buoy data:', {
          count: data.count,
          successful: data.successful,
          failed: data.failed,
          networks: data.networks,
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
  }, [apiBaseUrl, includeSpectral, showNetworkFilter])

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
      if (buoy.status === 'success' && buoy.observation) {
        const height = buoy.observation.wave_height_m
        if (height && height > maxWaveHeight) {
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
        const normalized = maxWaveHeight > 0 ? waveHeight / maxWaveHeight : 0
        const color = waveHeightToColor(normalized)

        // Draw circle - different shapes for different networks
        ctx.fillStyle = color
        ctx.beginPath()

        if (buoy.network === 'CDIP') {
          // Diamond shape for CDIP
          ctx.moveTo(x, y - 8)
          ctx.lineTo(x + 8, y)
          ctx.lineTo(x, y + 8)
          ctx.lineTo(x - 8, y)
          ctx.closePath()
        } else {
          // Circle for NDBC
          ctx.arc(x, y, 8, 0, 2 * Math.PI)
        }
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
        // Draw gray marker for failed buoys
        ctx.fillStyle = 'rgba(150, 150, 150, 0.5)'
        ctx.beginPath()
        if (buoy.network === 'CDIP') {
          ctx.moveTo(x, y - 6)
          ctx.lineTo(x + 6, y)
          ctx.lineTo(x, y + 6)
          ctx.lineTo(x - 6, y)
          ctx.closePath()
        } else {
          ctx.arc(x, y, 6, 0, 2 * Math.PI)
        }
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

  // Render swell component
  const renderSwellComponent = (label: string, swell: SwellComponent | undefined) => {
    if (!swell || swell.height_m === null) return null

    return (
      <div className="bg-slate-700/50 rounded p-2 text-xs">
        <div className="font-semibold text-slate-200 mb-1">{label}</div>
        <div className="grid grid-cols-2 gap-1 text-slate-300">
          <span>Height:</span>
          <span className="font-mono">{swell.height_ft?.toFixed(1)} ft ({swell.height_m?.toFixed(2)} m)</span>
          {swell.period_s && (
            <>
              <span>Period:</span>
              <span className="font-mono">{swell.period_s?.toFixed(1)} s</span>
            </>
          )}
          {swell.direction_deg && (
            <>
              <span>Direction:</span>
              <span className="font-mono">{swell.direction_deg?.toFixed(0)}° {swell.direction_cardinal}</span>
            </>
          )}
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-900">
        <div className="text-center">
          <div className="text-white text-xl mb-2">Loading Buoy Data...</div>
          <div className="text-slate-400 text-sm">Fetching NDBC & CDIP observations</div>
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
      {/* Controls */}
      <div className="mb-4 flex gap-4 items-center">
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={showNetworkFilter.ndbc}
            onChange={(e) => setShowNetworkFilter(prev => ({ ...prev, ndbc: e.target.checked }))}
            className="rounded"
          />
          NDBC
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={showNetworkFilter.cdip}
            onChange={(e) => setShowNetworkFilter(prev => ({ ...prev, cdip: e.target.checked }))}
            className="rounded"
          />
          CDIP
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={includeSpectral}
            onChange={(e) => setIncludeSpectral(e.target.checked)}
            className="rounded"
          />
          Include Spectral Data
        </label>
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
          Real-Time Buoy Observations - California
        </h3>

        <div className="grid grid-cols-2 gap-4 text-sm text-slate-300 mb-4">
          <div>
            <span className="font-semibold">Latitude:</span> {buoyData.bounds.min_lat.toFixed(2)}°N to {buoyData.bounds.max_lat.toFixed(2)}°N
          </div>
          <div>
            <span className="font-semibold">Longitude:</span> {buoyData.bounds.min_lon.toFixed(2)}°W to {buoyData.bounds.max_lon.toFixed(2)}°W
          </div>
          <div>
            <span className="font-semibold">Total Buoys:</span> {buoyData.count} (NDBC: {buoyData.networks.ndbc}, CDIP: {buoyData.networks.cdip})
          </div>
          <div>
            <span className="font-semibold">Reporting:</span> {buoyData.successful} / {buoyData.count}
          </div>
        </div>

        {/* Legend */}
        <div className="border-t border-slate-700 pt-3">
          <div className="font-semibold text-white text-sm mb-2">Legend</div>
          <div className="flex items-center gap-4 text-xs text-slate-400 mb-2">
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded-full bg-cyan-400 border border-white"></div>
              <span>NDBC Buoy</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rotate-45 bg-cyan-400 border border-white"></div>
              <span>CDIP Buoy</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded-full bg-gray-500/50"></div>
              <span>No Data</span>
            </div>
          </div>
        </div>

        {/* Color Legend */}
        <div className="border-t border-slate-700 pt-3">
          <div className="font-semibold text-white text-sm mb-2">Significant Wave Height</div>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <div className="flex-1 h-6 rounded" style={{
              background: 'linear-gradient(to right, rgb(0,150,255), rgb(100,250,255), rgb(255,255,100), rgb(255,150,0), rgb(255,0,0))'
            }} />
          </div>
          <div className="flex justify-between w-full text-xs text-slate-400 mt-1">
            <span>0 ft</span>
            <span>Small</span>
            <span>Moderate</span>
            <span>Large</span>
            <span>Very Large</span>
          </div>
        </div>

        {/* Buoy Details */}
        <div className="border-t border-slate-700 pt-3 mt-3">
          <div className="font-semibold text-white text-sm mb-2">Buoy Details (sorted by wave height)</div>
          <div className="max-h-64 overflow-y-auto space-y-2 text-xs">
            {buoyData.buoys
              .filter(b => b.status === 'success' && b.observation?.wave_height_m !== null)
              .sort((a, b) => (b.observation?.wave_height_m || 0) - (a.observation?.wave_height_m || 0))
              .map(buoy => (
                <div
                  key={buoy.station_id}
                  className={`p-2 rounded cursor-pointer transition-colors ${
                    selectedBuoy?.station_id === buoy.station_id
                      ? 'bg-slate-600'
                      : 'bg-slate-700/50 hover:bg-slate-700'
                  }`}
                  onClick={() => setSelectedBuoy(selectedBuoy?.station_id === buoy.station_id ? null : buoy)}
                >
                  <div className="flex justify-between text-slate-300">
                    <span>
                      <span className={`inline-block w-2 h-2 mr-1 ${buoy.network === 'CDIP' ? 'rotate-45' : 'rounded-full'}`}
                        style={{ backgroundColor: waveHeightToColor(buoy.observation.wave_height_m / 5) }}
                      ></span>
                      {buoy.name} ({buoy.station_id})
                      <span className="text-slate-500 ml-1">[{buoy.network}]</span>
                    </span>
                    <span className="font-mono">
                      {buoy.observation.wave_height_ft?.toFixed(1)} ft
                      ({buoy.observation.wave_height_m?.toFixed(2)} m)
                    </span>
                  </div>

                  {/* Expanded details when selected */}
                  {selectedBuoy?.station_id === buoy.station_id && (
                    <div className="mt-2 pt-2 border-t border-slate-600">
                      <div className="grid grid-cols-2 gap-2 text-slate-400 mb-2">
                        <div>
                          <span className="text-slate-500">Period:</span>{' '}
                          {buoy.observation.dominant_wave_period_s?.toFixed(1) || buoy.observation.dominant_period_s?.toFixed(1) || '-'} s
                        </div>
                        <div>
                          <span className="text-slate-500">Direction:</span>{' '}
                          {buoy.observation.mean_wave_direction_deg?.toFixed(0) || buoy.observation.peak_direction_deg?.toFixed(0) || '-'}°
                        </div>
                        {buoy.observation.water_temp_c && (
                          <div>
                            <span className="text-slate-500">Water Temp:</span>{' '}
                            {buoy.observation.water_temp_c?.toFixed(1)}°C
                          </div>
                        )}
                        {buoy.depth_m && (
                          <div>
                            <span className="text-slate-500">Depth:</span>{' '}
                            {buoy.depth_m} m
                          </div>
                        )}
                      </div>

                      {/* Spectral/Partitioned Data */}
                      {buoy.spectral_data && (
                        <div className="mt-2 space-y-2">
                          <div className="text-slate-300 font-semibold text-xs">Partitioned Wave Data:</div>

                          {/* NDBC Swell/Wind Wave separation */}
                          {buoy.spectral_data.swell && (
                            <div className="grid grid-cols-2 gap-2">
                              {renderSwellComponent('Swell', buoy.spectral_data.swell)}
                              {renderSwellComponent('Wind Waves', buoy.spectral_data.wind_waves)}
                            </div>
                          )}

                          {/* CDIP Multi-partition data */}
                          {buoy.spectral_data.partitions && buoy.spectral_data.partitions.length > 0 && (
                            <div className="space-y-1">
                              {buoy.spectral_data.partitions.map((partition, idx) => (
                                <div key={idx} className="bg-slate-700/50 rounded p-2">
                                  <div className="flex justify-between text-slate-300">
                                    <span className="font-semibold">
                                      Swell {partition.partition_id}
                                      <span className="text-slate-500 font-normal ml-1">
                                        ({partition.type.replace(/_/g, ' ')})
                                      </span>
                                    </span>
                                    <span className="font-mono">
                                      {partition.height_ft?.toFixed(1)} ft @ {partition.period_s?.toFixed(1)}s
                                      {partition.direction_deg && ` from ${partition.direction_deg.toFixed(0)}°`}
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Combined stats */}
                          {buoy.spectral_data.combined && (
                            <div className="text-slate-500 text-xs mt-1">
                              Combined: {buoy.spectral_data.combined.significant_height_ft?.toFixed(1)} ft,
                              Avg Period: {buoy.spectral_data.combined.average_period_s?.toFixed(1)}s
                              {buoy.spectral_data.combined.steepness && `, Steepness: ${buoy.spectral_data.combined.steepness}`}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Link to fetch spectral data if not loaded */}
                      {!buoy.spectral_data && !includeSpectral && (
                        <div className="text-slate-500 text-xs mt-1">
                          Enable "Include Spectral Data" to see swell partitions
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}
