/**
 * API Types
 *
 * These types should match the Pydantic models in backend/api/app/models/
 * Consider auto-generating these from OpenAPI schema
 */

export interface Spot {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  timezone: string;
  description?: string;
}

export interface ForecastPeriod {
  timestamp: string;
  wave_height: number;
  wave_period: number;
  swell_direction: number;
  wind_speed: number;
  wind_direction: number;
  rating: number;  // 1-10
  confidence: number;  // 0-1
}

export interface ForecastResponse {
  spot_id: string;
  generated_at: string;
  periods: ForecastPeriod[];
}

export interface SpotResponse {
  spot: Spot;
}

export interface SpotListResponse {
  spots: Spot[];
  total: number;
}
