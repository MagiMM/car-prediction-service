import json
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import price, transmission  # REGRESJA + KLASYFIKACJA

# Ścieżki
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# FastAPI app
app = FastAPI(
    title="Car Prediction Service",
    description="API do przewidywania ceny i typu skrzyni biegów",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Zmienne globalne dla modeli
app.state.price_model = None  # REGRESJA
app.state.transmission_model = None  # KLASYFIKACJA
app.state.models_metadata = None


@app.on_event("startup")
async def load_models():
    """Wczytanie modeli przy starcie aplikacji"""
    try:
        # Wczytanie modeli
        price_model_path = MODELS_DIR / "price_model.pkl"  # REGRESJA
        transmission_model_path = MODELS_DIR / "transmission_model.pkl"  # KLASYFIKACJA
        metadata_path = MODELS_DIR / "models_metadata.json"

        app.state.price_model = joblib.load(price_model_path)  # REGRESJA
        app.state.transmission_model = joblib.load(
            transmission_model_path
        )  # KLASYFIKACJA

        # Wczytanie metadata
        with open(metadata_path, "r") as f:
            app.state.models_metadata = json.load(f)

        print("Models loaded successfully")
        print(f"   - Price model: {price_model_path}")  # REGRESJA
        print(f"   - Transmission model: {transmission_model_path}")  # KLASYFIKACJA
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Warning: Some models may not be available")
        # Nie rzucamy wyjątku - API może działać z tylko jednym modelem


@app.get("/", tags=["General"])
def read_root():
    """Root endpoint z informacjami o API"""
    return {
        "message": "Car Prediction Service API",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict_price": "/api/predict-price",  # REGRESJA
            "predict_transmission": "/api/predict-transmission",  # KLASYFIKACJA
        },
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint"""
    price_model_loaded = app.state.price_model is not None
    transmission_model_loaded = app.state.transmission_model is not None
    models_loaded = price_model_loaded or transmission_model_loaded

    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "price_model": price_model_loaded,  # REGRESJA
        "transmission_model": transmission_model_loaded,  # KLASYFIKACJA
    }


# Include routers
app.include_router(price.router, prefix="/api", tags=["Price Prediction"])  # REGRESJA
app.include_router(
    transmission.router, prefix="/api", tags=["Transmission Classification"]
)  # KLASYFIKACJA
