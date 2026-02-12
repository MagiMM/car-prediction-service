from pydantic import BaseModel, Field, field_validator
from typing import Literal


# Używane zarówno przez REGRESJA jak i KLASYFIKACJA
class CarInput(BaseModel):
    """Model wejściowy dla danych o samochodzie"""
    make: Literal['Honda', 'Ford', 'BMW', 'Audi', 'Toyota'] = Field(
        ...,
        description="Marka samochodu",
        examples=["Honda"]
    )
    model: Literal['Model A', 'Model B', 'Model C', 'Model D', 'Model E'] = Field(
        ...,
        description="Model samochodu",
        examples=["Model B"]
    )
    year: int = Field(
        ...,
        ge=2000,
        le=2026,
        description="Rok produkcji",
        examples=[2020]
    )
    engine_size: float = Field(
        ...,
        ge=1.0,
        le=4.5,
        description="Pojemność silnika w litrach",
        examples=[2.5]
    )
    mileage: int = Field(
        ...,
        ge=0,
        description="Przebieg w kilometrach",
        examples=[50000]
    )
    fuel_type: Literal['Petrol', 'Diesel', 'Electric'] = Field(
        ...,
        description="Rodzaj paliwa",
        examples=["Petrol"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "make": "Honda",
                "model": "Model B",
                "year": 2020,
                "engine_size": 2.5,
                "mileage": 50000,
                "fuel_type": "Petrol"
            }
        }


# ============================================================
# REGRESJA: Response model
# ============================================================
class PricePredictionResponse(BaseModel):
    """Odpowiedź z predykcją ceny"""
    predicted_price: float = Field(
        ...,
        description="Przewidywana cena w PLN"
    )
    input_data: CarInput = Field(
        ...,
        description="Dane wejściowe użyte do predykcji"
    )
    model_info: dict = Field(
        ...,
        description="Informacje o modelu"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 30797.46,
                "input_data": {
                    "make": "Honda",
                    "model": "Model B",
                    "year": 2020,
                    "engine_size": 2.5,
                    "mileage": 50000,
                    "fuel_type": "Petrol"
                },
                "model_info": {
                    "model_type": "RandomForestRegressor",
                    "r2_score": 0.7915
                }
            }
        }