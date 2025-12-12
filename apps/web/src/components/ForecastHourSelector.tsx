'use client'

interface ForecastHourSelectorProps {
  value: number
  onChange: (hour: number) => void
  maxHours?: number  // 384 for GFS/WW3, 192 for RTOFS
  disabled?: boolean
}

// Format forecast hour for display
function formatForecastHour(hour: number): string {
  if (hour === 0) return 'Now'
  if (hour < 24) return `+${hour}h`
  const days = Math.floor(hour / 24)
  const remainingHours = hour % 24
  if (remainingHours === 0) return `+${days}d`
  return `+${days}d ${remainingHours}h`
}

// Generate forecast hour options based on the configured intervals
// 0-48 @ 3hr, then 72+ @ 24hr
function generateForecastHours(maxHours: number): number[] {
  const hours: number[] = []

  // 3-hourly for first 48 hours
  for (let h = 0; h <= 48 && h <= maxHours; h += 3) {
    hours.push(h)
  }

  // 24-hourly from 72 hours onwards
  for (let h = 72; h <= maxHours; h += 24) {
    hours.push(h)
  }

  return hours
}

export default function ForecastHourSelector({
  value,
  onChange,
  maxHours = 384,
  disabled = false
}: ForecastHourSelectorProps) {
  const forecastHours = generateForecastHours(maxHours)

  return (
    <div className="flex items-center gap-2">
      <label className="text-sm font-medium text-slate-300">
        Forecast:
      </label>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="bg-slate-700 text-white text-sm rounded px-3 py-1.5 border border-slate-600
                   focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                   disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {forecastHours.map((hour) => (
          <option key={hour} value={hour}>
            {formatForecastHour(hour)}
          </option>
        ))}
      </select>
    </div>
  )
}

export { generateForecastHours, formatForecastHour }
