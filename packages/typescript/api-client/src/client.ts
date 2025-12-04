/**
 * API Client
 */

import axios, { AxiosInstance } from 'axios';
import type { ForecastResponse, SpotResponse, SpotListResponse } from './types';

export class WaveForecastClient {
  private client: AxiosInstance;

  constructor(baseURL: string, apiKey?: string) {
    this.client = axios.create({
      baseURL,
      headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {},
    });
  }

  /**
   * Get forecast for a spot
   */
  async getForecast(spotId: string): Promise<ForecastResponse> {
    const response = await this.client.get(`/forecast/${spotId}`);
    return response.data;
  }

  /**
   * Get spot details
   */
  async getSpot(spotId: string): Promise<SpotResponse> {
    const response = await this.client.get(`/spots/${spotId}`);
    return response.data;
  }

  /**
   * List all spots
   */
  async listSpots(): Promise<SpotListResponse> {
    const response = await this.client.get('/spots');
    return response.data;
  }

  /**
   * Search spots near coordinates
   */
  async searchSpots(latitude: number, longitude: number, radiusKm: number = 50): Promise<SpotListResponse> {
    const response = await this.client.get('/spots/search', {
      params: { latitude, longitude, radius: radiusKm }
    });
    return response.data;
  }
}
