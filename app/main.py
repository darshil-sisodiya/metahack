"""OpenEnv API - Main FastAPI Application"""

from fastapi import FastAPI

app = FastAPI(
    title="OpenEnv",
    description="OpenEnv Environment API",
    version="0.1.0",
)


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
