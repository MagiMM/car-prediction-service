from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import joblib
import json

from app.routers import price, category

# Ścieżki
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# FastAPI app
app = FastAPI(
    title="Car Prediction Service",
    description="API do przewidywania ceny i typu skrzyni biegów samochodów",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
app.state.price_model = None
app.state.transmission_model = None
app.state.models_metadata = None


@app.on_event("startup")
async def load_models():
    """Wczytanie modeli przy starcie aplikacji"""
    try:
        # Wczytanie modeli
        price_model_path = MODELS_DIR / "price_model.pkl"
        transmission_model_path = MODELS_DIR / "transmission_model.pkl"
        metadata_path = MODELS_DIR / "models_metadata.json"
        
        app.state.price_model = joblib.load(price_model_path)
        app.state.transmission_model = joblib.load(transmission_model_path)
        
        # Wczytanie metadata
        with open(metadata_path, 'r') as f:
            app.state.models_metadata = json.load(f)
        
        print("✅ Modele wczytane pomyślnie")
        print(f"   - Price model: {price_model_path}")
        print(f"   - Transmission model: {transmission_model_path}")
    except Exception as e:
        print(f"❌ Błąd wczytywania modeli: {e}")
        raise


@app.get("/", tags=["General"])
def read_root():
    """Root endpoint z informacjami o API"""
    return {
        "message": "Car Prediction Service API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict_price": "/predict-price",
            "predict_transmission": "/predict-transmission"
        }
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint"""
    models_loaded = (
        app.state.price_model is not None and 
        app.state.transmission_model is not None
    )
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "price_model": app.state.price_model is not None,
        "transmission_model": app.state.transmission_model is not None
    }


# Include routers
app.include_router(price.router, prefix="/api", tags=["Price Prediction"])
app.include_router(category.router, prefix="/api", tags=["Transmission Prediction"])