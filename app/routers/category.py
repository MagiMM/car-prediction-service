# ============================================================
# KLASYFIKACJA: Endpoint for transmission prediction
# ============================================================
from fastapi import APIRouter, HTTPException, Request
import pandas as pd
import numpy as np

from app.schemas import CarInput, TransmissionPredictionResponse

router = APIRouter()


# KLASYFIKACJA: Prediction endpoint
@router.post("/predict-transmission", response_model=TransmissionPredictionResponse)
async def predict_transmission(car: CarInput, request: Request):
    """
    Przewidywanie typu skrzyni biegów (Manual/Automatic) na podstawie parametrów samochodu
    
    **Parametry:**
    - make: Marka samochodu (Honda, Ford, BMW, Audi, Toyota)
    - model: Model samochodu (Model A/B/C/D/E)
    - year: Rok produkcji (2000-2026)
    - engine_size: Pojemność silnika w litrach (1.0-4.5)
    - mileage: Przebieg w kilometrach (>= 0)
    - fuel_type: Rodzaj paliwa (Petrol, Diesel, Electric)
    
    **Zwraca:**
    - predicted_transmission: Manual lub Automatic
    - probability: Prawdopodobieństwa dla każdej klasy
    - input_data: Dane wejściowe
    - model_info: Informacje o modelu ML
    
    **Uwaga:** Model ma niską dokładność (47.5%), więc predykcje mogą być nieprecyzyjne.
    """
    try:
        # KLASYFIKACJA: Pobranie modelu z app state
        model = request.app.state.transmission_model
        metadata = request.app.state.models_metadata.get('transmission_model', {})
        
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model nie jest dostępny. Sprawdź czy serwis został poprawnie uruchomiony."
            )
        
        # Przygotowanie danych do predykcji
        input_data = pd.DataFrame([{
            'Make': car.make,
            'Model': car.model,
            'Year': car.year,
            'Engine Size': car.engine_size,
            'Mileage': car.mileage,
            'Fuel Type': car.fuel_type
        }])
        
        # KLASYFIKACJA: Predykcja
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # KLASYFIKACJA: Mapowanie prawdopodobieństw do klas
        classes = model.classes_
        probability_dict = {
            str(classes[0]): round(float(probabilities[0]), 4),
            str(classes[1]): round(float(probabilities[1]), 4)
        }
        
        # Zwrócenie wyniku
        return TransmissionPredictionResponse(
            predicted_transmission=str(prediction),
            probability=probability_dict,
            input_data=car,
            model_info={
                "model_type": metadata.get('type', 'LogisticRegression'),
                "accuracy": metadata.get('metrics', {}).get('accuracy', 0.0),
                "f1_score": metadata.get('metrics', {}).get('f1_score', 0.0),
                "warning": "Model ma niską dokładność (47.5%). Predykcje mogą być nieprecyzyjne."
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Błąd podczas predykcji: {str(e)}"
        )