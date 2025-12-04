# @wave-forecast/api-client

Shared TypeScript API client for Wave Forecast API.

## Usage

### In Web App

```typescript
import { WaveForecastClient } from '@wave-forecast/api-client';

const client = new WaveForecastClient(
  process.env.NEXT_PUBLIC_API_URL!
);

const forecast = await client.getForecast('ocean-beach-sf');
```

### In Mobile App

```typescript
import { WaveForecastClient } from '@wave-forecast/api-client';
import Constants from 'expo-constants';

const client = new WaveForecastClient(
  Constants.expoConfig?.extra?.apiUrl
);

const spots = await client.searchSpots(37.7749, -122.4194);
```

## Type Generation

In the future, consider generating these types automatically from the FastAPI OpenAPI schema:

```bash
# Using openapi-typescript
npx openapi-typescript http://localhost:8000/openapi.json -o src/types.ts
```

## Development

```bash
pnpm install
pnpm type-check
pnpm lint
```
