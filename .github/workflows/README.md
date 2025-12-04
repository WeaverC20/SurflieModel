# GitHub Workflows

CI/CD automation for the Wave Forecast project.

## Workflows

### `test.yml`
Runs on every push and PR:
- Python tests, linting, type checking
- TypeScript tests, linting, type checking

### `deploy-web.yml`
Deploys web app on push to main:
- Builds Next.js app
- Note: Vercel typically handles this automatically

### `scheduled-data-fetch.yml`
Scheduled data fetching:
- Runs every 6 hours
- Fetches NOAA and buoy data
- Can also be triggered manually

## Secrets Required

Configure these in GitHub Settings > Secrets:

- `NOAA_API_KEY` - NOAA API key (if required)
- `DATABASE_URL` - PostgreSQL connection string
- `AWS_ACCESS_KEY_ID` - For S3 model storage
- `AWS_SECRET_ACCESS_KEY` - For S3 model storage
- `RAILWAY_TOKEN` - For backend deployment (if using Railway)

## Future Workflows

Consider adding:
- `deploy-backend.yml` - Deploy FastAPI to Railway/Fly.io
- `deploy-mobile.yml` - Build and submit mobile app with EAS
- `train-model.yml` - Trigger model training
- `benchmark.yml` - Performance benchmarks
