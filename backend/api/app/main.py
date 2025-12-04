"""FastAPI application entry point"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Wave Forecast API",
    description="API for wave forecasting and surf condition predictions",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Wave Forecast API", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# TODO: Add routers
# from app.routers import forecast, spots, users
# app.include_router(forecast.router, prefix="/api/v1/forecast", tags=["forecast"])
# app.include_router(spots.router, prefix="/api/v1/spots", tags=["spots"])
# app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
