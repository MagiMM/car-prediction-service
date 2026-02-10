# ============================================================
# REGRESJA: Endpoint for price prediction
# ============================================================
from fastapi import APIRouter, HTTPException, Request
import pandas as pd

from app.schemas import CarInput, PricePredictionResponse

router = APIRouter()


# REGRESJA: Prediction endpoint
@router.post("/predict-price", response_model=PricePredictionResponse)
async def predict_price(car: CarInput, request: Request):
    """
    Przewidywanie ceny samochodu na podstawie parametrów
    
    **Parametry:**
    - make: Marka samochodu (Honda, Ford, BMW, Audi, Toyota)
    - model: Model samochodu (Model A/B/C/D/E)
    - year: Rok produkcji (2000-2026)
    - engine_size: Pojemność silnika w litrach (1.0-4.5)
    - mileage: Przebieg w kilometrach (>= 0)
    - fuel_type: Rodzaj paliwa (Petrol, Diesel, Electric)
    
    **Zwraca:**
    - predicted_price: Przewidywana cena w PLN
    - input_data: Dane wejściowe
    - model_info: Informacje o modelu ML
    """
    try:
        # REGRESJA: Pobranie modelu z app state
        model = request.app.state.price_model
        metadata = request.app.state.models_metadata.get('price_model', {})
        
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model nie jest dostępny. Sprawdź czy serwis został poprawnie uruchomiony."
            )
        
        # REGRESJA: Przygotowanie danych do predykcji
        # Uwaga: nazwy kolumn muszą być takie jak podczas treningu
        input_data = pd.DataFrame([{
            'Make': car.make,
            'Model': car.model,
            'Year': car.year,
            'Engine Size': car.engine_size,
            'Mileage': car.mileage,
            'Fuel Type': car.fuel_type
        }])
        
        # REGRESJA: Predykcja
        prediction = model.predict(input_data)[0]
        
        # Zwrócenie wyniku
        return PricePredictionResponse(
            predicted_price=round(float(prediction), 2),
            input_data=car,
            model_info={
                "model_type": metadata.get('type', 'RandomForestRegressor'),
                "r2_score": metadata.get('metrics', {}).get('r2_score', 0.0),
                "mae": metadata.get('metrics', {}).get('mae', 0.0)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Błąd podczas predykcji: {str(e)}"
        )