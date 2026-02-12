# ============================================================
# CLASSIFICATION: Endpoint for transmission prediction
# ============================================================
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.schemas import CarInputWithPrice, TransmissionPredictionResponse

router = APIRouter()


@router.post("/predict-transmission", response_model=TransmissionPredictionResponse)
async def predict_transmission(car: CarInputWithPrice, request: Request):
    """
    Przewidywanie typu skrzyni biegów na podstawie parametrów samochodu

    **Parametry:**
    - make: Marka samochodu (Honda, Ford, BMW, Audi, Toyota)
    - model: Model samochodu (Model A/B/C/D/E)
    - year: Rok produkcji (2000-2026)
    - engine_size: Pojemność silnika w litrach (1.0-4.5)
    - mileage: Przebieg w kilometrach (>= 0)
    - fuel_type: Rodzaj paliwa (Petrol, Diesel, Electric)
    - price: Cena samochodu w PLN (>= 0)

    **Zwraca:**
    - predicted_transmission: Przewidywany typ skrzyni biegów (Manual/Automatic)
    - probability: Prawdopodobieństwo dla każdej klasy
    - confidence: Poziom pewności predykcji
    - input_data: Dane wejściowe
    - model_info: Informacje o modelu ML
    """
    try:
        # Pobranie modelu z app state
        model = request.app.state.transmission_model
        metadata = request.app.state.models_metadata.get("transmission_model", {})

        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model klasyfikacji skrzyni biegów nie jest dostępny. Sprawdź czy serwis został poprawnie uruchomiony.",
            )

        # Feature engineering - przygotowanie dodatkowych cech
        car_age = 2026 - car.year
        mileage_per_year = car.mileage / car_age if car_age > 0 else car.mileage

        # Engine category
        if car.engine_size <= 1.5:
            engine_category = "Small"
        elif car.engine_size <= 2.5:
            engine_category = "Medium"
        elif car.engine_size <= 3.5:
            engine_category = "Large"
        else:
            engine_category = "Very_Large"

        # Price category - używamy stałych wartości na podstawie analizy danych
        if car.price <= 20000:
            price_category = "Low"
        elif car.price <= 25000:
            price_category = "Medium"
        elif car.price <= 35000:
            price_category = "High"
        else:
            price_category = "Premium"

        # Przygotowanie danych do predykcji - muszą być zgodne z cechami z treningu
        input_data = pd.DataFrame(
            [
                {
                    "Make": car.make,
                    "Model": car.model,
                    "Year": car.year,
                    "Engine Size": car.engine_size,
                    "Mileage": car.mileage,
                    "Fuel Type": car.fuel_type,
                    "Price": car.price,
                    "Car_Age": car_age,
                    "Mileage_Per_Year": mileage_per_year,
                    "Engine_Category": engine_category,
                    "Price_Category": price_category,
                }
            ]
        )

        # Predykcja
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Pobieranie nazw klas
        class_names = model.classes_
        prob_dict = {
            class_names[i]: round(float(probabilities[i]), 4)
            for i in range(len(class_names))
        }

        # Poziom pewności (najwyższe prawdopodobieństwo)
        confidence = round(float(max(probabilities)), 4)

        # Informacje o modelu
        model_info = {
            "model_type": metadata.get("type", "SVM"),
            "accuracy": metadata.get("metrics", {}).get("accuracy", 0.54),
            "feature_count": metadata.get("n_features", 11),
            "classes": metadata.get("classes", ["Automatic", "Manual"]),
        }

        return TransmissionPredictionResponse(
            predicted_transmission=prediction,
            probability=prob_dict,
            confidence=confidence,
            input_data=car,
            model_info=model_info,
        )

    except Exception as e:
        # Logowanie błędu dla debugowania
        print(f"Error in transmission prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Wystąpił błąd podczas przewidywania typu skrzyni biegów: {str(e)}",
        )
