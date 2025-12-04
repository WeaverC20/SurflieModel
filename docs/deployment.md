# Deployment Guide

## Overview

This guide covers deploying all components of the Wave Forecast app.

## Prerequisites

- Git repository on GitHub
- NOAA API access (if required)
- AWS account (for S3 model storage)
- Trained ML model uploaded to S3

## Web App (Vercel)

### Initial Setup

1. **Connect to Vercel**:
   ```bash
   # Install Vercel CLI
   npm install -g vercel

   # Login
   vercel login

   # Deploy
   cd apps/web
   vercel
   ```

2. **Or use Vercel Dashboard**:
   - Visit https://vercel.com/new
   - Import GitHub repository
   - Set root directory: `apps/web`
   - Framework preset: Next.js
   - Auto-detected, no config needed

3. **Environment Variables** (in Vercel Dashboard):
   ```
   NEXT_PUBLIC_API_URL=https://your-api.railway.app/api/v1
   NEXT_PUBLIC_MAPBOX_TOKEN=your-mapbox-token
   ```

4. **Deploy**:
   - Push to `main` branch
   - Vercel auto-deploys

### Custom Domain

1. Add domain in Vercel dashboard
2. Update DNS records as instructed
3. SSL certificate auto-generated

## Backend API (Railway)

### Initial Setup

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Create new project**:
   ```bash
   railway init
   ```

3. **Deploy API service**:
   ```bash
   # From root directory
   railway up
   ```

4. **Configure Build**:
   - In Railway dashboard → Settings:
     - Root Directory: `backend/api`
     - Build Command: `pip install -r requirements.txt && pip install -e ../../packages/python/common && pip install -e ../../ml/inference`
     - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

5. **Environment Variables** (Railway Dashboard):
   ```
   DATABASE_URL=postgresql://...  (auto-provided by Railway)
   REDIS_URL=redis://...  (add Redis plugin)
   SECRET_KEY=your-secret-key-here
   ENVIRONMENT=production
   NOAA_API_KEY=your-noaa-key
   NOAA_BASE_URL=https://api.weather.gov
   NDBC_BASE_URL=https://www.ndbc.noaa.gov
   MODEL_URL=s3://your-bucket/models/wave_model_v1.pt
   AWS_ACCESS_KEY_ID=your-aws-key
   AWS_SECRET_ACCESS_KEY=your-aws-secret
   AWS_REGION=us-west-2
   ```

6. **Add Plugins**:
   - PostgreSQL (auto-configures DATABASE_URL)
   - Redis (auto-configures REDIS_URL)

### Alternative: Fly.io

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Initialize
cd backend/api
fly launch

# Deploy
fly deploy
```

## Worker Service (Railway)

1. **Add new service** in Railway dashboard

2. **Configure**:
   - Root Directory: `backend/worker`
   - Build Command: Same as API
   - Start Command: `python scheduler.py`

3. **Environment Variables**: Same as API service

4. **Note**: Worker and API can share the same database and Redis

## Database (Railway PostgreSQL)

### Auto-Provisioned

- Railway automatically provisions PostgreSQL
- Connection string in `DATABASE_URL`

### Manual Setup (if needed)

1. **Create tables** (first deploy):
   ```bash
   # Using Alembic migrations
   railway run alembic upgrade head
   ```

2. **Seed data** (optional):
   ```bash
   railway run python scripts/seed_spots.py
   ```

### Backups

- Railway auto-backups daily
- Manual backup:
  ```bash
  railway run pg_dump $DATABASE_URL > backup.sql
  ```

## ML Models (S3)

### Upload Model

```bash
# After training
cd ml/training
aws s3 cp artifacts/wave_model_v1.pt s3://your-bucket/models/wave_model_v1.pt

# Set public read permissions (if needed)
aws s3api put-object-acl \
  --bucket your-bucket \
  --key models/wave_model_v1.pt \
  --acl public-read
```

### Model Versioning

```
s3://your-bucket/models/
├── wave_model_v1.pt
├── wave_model_v2.pt
└── wave_model_latest.pt  # Symlink or copy
```

### Update Backend

Update `MODEL_URL` environment variable in Railway:
```
MODEL_URL=s3://your-bucket/models/wave_model_v2.pt
```

Redeploy or restart service to load new model.

## Mobile App (Expo EAS)

### Initial Setup

1. **Install EAS CLI**:
   ```bash
   npm install -g eas-cli
   eas login
   ```

2. **Configure EAS**:
   ```bash
   cd apps/mobile
   eas build:configure
   ```

3. **Update app.json**:
   ```json
   {
     "expo": {
       "extra": {
         "eas": {
           "projectId": "your-project-id"
         }
       }
     }
   }
   ```

### Build

```bash
# iOS
eas build --platform ios --profile production

# Android
eas build --platform android --profile production

# Both
eas build --platform all --profile production
```

### Submit to Stores

```bash
# iOS App Store
eas submit --platform ios

# Google Play
eas submit --platform android
```

### Over-The-Air (OTA) Updates

```bash
# Publish update (no rebuild required for JS changes)
eas update --branch production --message "Fix forecast display bug"
```

## GitHub Actions (CI/CD)

### Secrets Configuration

In GitHub repository settings → Secrets and variables → Actions, add:

```
NOAA_API_KEY
DATABASE_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
RAILWAY_TOKEN  (from Railway dashboard)
```

### Workflows

Already configured in `.github/workflows/`:
- `test.yml` - Runs on every push/PR
- `deploy-web.yml` - Vercel handles this automatically
- `scheduled-data-fetch.yml` - Runs every 6 hours

## Monitoring

### Railway Logs

```bash
# View API logs
railway logs

# Follow logs
railway logs -f
```

### Vercel Logs

- Dashboard → Deployments → Logs
- Or use Vercel CLI: `vercel logs`

### Error Tracking (Future)

Consider adding:
- Sentry for error tracking
- LogRocket for session replay
- Datadog for monitoring

## Rollback Strategy

### API/Worker (Railway)

1. **Railway Dashboard**:
   - Deployments → Select previous deployment
   - Click "Redeploy"

2. **Or via CLI**:
   ```bash
   railway rollback
   ```

### Web (Vercel)

- Dashboard → Deployments
- Click on previous successful deployment
- Click "Promote to Production"

### Mobile (Expo)

```bash
# Rollback OTA update
eas update:rollback --branch production
```

For native changes, need to submit new build.

### Database (Dangerous!)

```bash
# Restore from backup
railway run psql $DATABASE_URL < backup.sql
```

## Performance Optimization

### API
- Enable Redis caching (15-30 min TTL)
- Use connection pooling
- Enable gzip compression

### Database
- Add indexes on frequently queried columns
- Use read replicas for scaling
- Enable query caching

### Frontend
- Vercel Edge caching
- Image optimization (next/image)
- Code splitting

## Health Checks

### API Health Check

```bash
curl https://your-api.railway.app/health
```

Expected response:
```json
{"status": "healthy"}
```

### Monitor from Worker

```python
# In backend/worker/tasks/health_check.py
import httpx

async def check_api_health():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://your-api.railway.app/health")
        if response.status_code != 200:
            # Send alert (email, Slack, etc.)
            pass
```

## Cost Estimates (Approximate)

### MVP (Low Traffic)
- Vercel: Free tier (likely sufficient)
- Railway: ~$10-20/month (API + Worker + DB + Redis)
- AWS S3: ~$1-5/month (model storage)
- Expo EAS: Free for personal projects
- **Total**: ~$11-25/month

### Growth (Moderate Traffic)
- Vercel: Free or ~$20/month
- Railway: ~$50-100/month (scaled instances)
- AWS: ~$10/month
- Expo: ~$29/month (paid plan)
- **Total**: ~$109-159/month

## Checklist Before Going Live

- [ ] Environment variables set in all services
- [ ] Database migrations run
- [ ] ML model uploaded to S3 and accessible
- [ ] API health check passing
- [ ] Web app builds successfully
- [ ] Mobile app tested on iOS and Android
- [ ] CORS configured correctly
- [ ] Rate limiting enabled (future)
- [ ] Error tracking set up
- [ ] Monitoring dashboards configured
- [ ] Backup strategy tested
- [ ] Domain configured with SSL
- [ ] Terms of Service and Privacy Policy (if required)

## Support & Troubleshooting

### Common Issues

**"Module not found" in Railway**:
- Check build command includes all package installations
- Verify root directory setting

**"Model not loading" error**:
- Verify S3 URL is correct
- Check AWS credentials
- Ensure model file exists

**Database connection errors**:
- Check DATABASE_URL format
- Verify network access (Railway handles this)
- Connection pool exhausted? Increase pool size

**Frontend can't reach API**:
- CORS configuration
- Check NEXT_PUBLIC_API_URL is set
- Verify API is deployed and healthy

### Getting Help

- Railway Discord: https://discord.gg/railway
- Vercel Discord: https://discord.gg/vercel
- Expo Discord: https://discord.gg/expo
