"""FastAPI application for the FPL Season Visualizer."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fpl_rl.data.downloader import DEFAULT_DATA_DIR
from fpl_rl.utils.constants import AVAILABLE_SEASONS

from .schemas import SimulationResponse
from .simulator import simulate_season

app = FastAPI(title="FPL Season Visualizer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration via environment variables
MODEL_PATH = os.environ.get("FPL_MODEL_PATH", "runs/fpl_ppo/best_model/best_model")
DATA_DIR = Path(os.environ.get("FPL_DATA_DIR", str(DEFAULT_DATA_DIR)))
PREDICTOR_DIR = os.environ.get("FPL_PREDICTOR_DIR")


@app.get("/api/seasons")
def get_seasons() -> dict:
    """Return list of available seasons."""
    return {"seasons": AVAILABLE_SEASONS}


@app.post("/api/simulate/{season}", response_model=SimulationResponse)
def simulate(season: str) -> SimulationResponse:
    """Simulate a full season using the trained RL model."""
    if season not in AVAILABLE_SEASONS:
        raise HTTPException(status_code=404, detail=f"Season {season} not available")

    predictor_dir = Path(PREDICTOR_DIR) if PREDICTOR_DIR else None
    try:
        result = simulate_season(
            season=season,
            model_path=MODEL_PATH,
            data_dir=DATA_DIR,
            predictor_model_dir=predictor_dir,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

    return result


# Serve frontend static files in production
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
