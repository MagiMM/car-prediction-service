# Car Prediction Service

API serwis do przewidywania ceny samochodów i typu skrzyni biegów przy użyciu Machine Learning.

## Opis projektu

Projekt implementuje dwa modele ML:
- **REGRESJA**: Przewidywanie ceny samochodu (RandomForestRegressor, R² = 79.2%)
- **KLASYFIKACJA**: Przewidywanie typu skrzyni biegów Manual/Automatic (LogisticRegression, Accuracy = 47.5%)

## Technologie

- **FastAPI** - REST API framework
- **scikit-learn** - modele Machine Learning
- **pandas** - przetwarzanie danych
- **Pydantic** - walidacja danych
- **uvicorn** - ASGI server

## Struktura projektu

```
car-prediction-service/
├── app/
│   ├── main.py              # Aplikacja FastAPI
│   ├── schemas.py           # Modele Pydantic
│   └── routers/
│       ├── price.py         # Endpoint predykcji ceny
│       └── category.py      # Endpoint predykcji transmisji
├── data/
│   └── car_pricing_amjad_zhour.csv
├── models/
│   ├── price_model.pkl
│   ├── transmission_model.pkl
│   └── models_metadata.json
├── scripts/
│   ├── 01_eda.py            # Eksploracja danych
│   └── 02_train_models.py   # Trening modeli
└── pyproject.toml
```

## Instalacja

### 1. Klonowanie repozytorium

```bash
git clone <repository-url>
cd car-prediction-service
```

### 2. Instalacja zależności

```bash
# Utwórz wirtualne środowisko
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# lub
.venv\Scripts\activate     # Windows

# Zainstaluj zależności
pip install -e .
```

lub z użyciem `uv`:

```bash
uv sync
```

### 3. Trening modeli

```bash
.venv/bin/python scripts/02_train_models.py
```

To wytrenuje modele i zapisze je w katalogu `models/`.

## Uruchomienie

### Serwer deweloperski

```bash
.venv/bin/uvicorn app.main:app --reload
```

Aplikacja będzie dostępna pod adresem: http://localhost:8000

### Dokumentacja API

Po uruchomieniu serwera, dokumentacja API jest dostępna pod:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Użycie API

### Health Check

```bash
curl http://localhost:8000/health
```

### Predykcja ceny

```bash
curl -X POST "http://localhost:8000/api/predict-price" \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Honda",
    "model": "Model B",
    "year": 2020,
    "engine_size": 2.5,
    "mileage": 50000,
    "fuel_type": "Petrol"
  }'
```

**Odpowiedź:**
```json
{
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
    "r2_score": 0.7915,
    "mae": 1937.17
  }
}
```

### Predykcja typu skrzyni biegów

```bash
curl -X POST "http://localhost:8000/api/predict-transmission" \
  -H "Content-Type: application/json" \
  -d '{
    "make": "BMW",
    "model": "Model A",
    "year": 2015,
    "engine_size": 3.0,
    "mileage": 80000,
    "fuel_type": "Diesel"
  }'
```

**Odpowiedź:**
```json
{
  "predicted_transmission": "Manual",
  "probability": {
    "Automatic": 0.4505,
    "Manual": 0.5495
  },
  "input_data": {
    "make": "BMW",
    "model": "Model A",
    "year": 2015,
    "engine_size": 3.0,
    "mileage": 80000,
    "fuel_type": "Diesel"
  },
  "model_info": {
    "model_type": "LogisticRegression",
    "accuracy": 0.475,
    "f1_score": 0.5116,
    "warning": "Model ma niską dokładność (47.5%). Predykcje mogą być nieprecyzyjne."
  }
}
```

## Parametry wejściowe

| Parametr | Typ | Zakres | Opis |
|----------|-----|--------|------|
| `make` | string | Honda, Ford, BMW, Audi, Toyota | Marka samochodu |
| `model` | string | Model A, B, C, D, E | Model samochodu |
| `year` | integer | 2000-2026 | Rok produkcji |
| `engine_size` | float | 1.0-4.5 | Pojemność silnika (litry) |
| `mileage` | integer | >= 0 | Przebieg (kilometry) |
| `fuel_type` | string | Petrol, Diesel, Electric | Rodzaj paliwa |

## Metryki modeli

### Model regresji (predykcja ceny)
- **R² Score**: 0.7915 (79.2%)
- **MAE**: 1937.17 PLN
- **RMSE**: 2388.70 PLN
- **Błąd względny**: 7.72%
- **Status**: Gotowy do produkcji

### Model klasyfikacji (predykcja transmisji)
- **Accuracy**: 0.4750 (47.5%)
- **Precision**: 0.4867
- **Recall**: 0.5392
- **F1-Score**: 0.5116
- **Status**: Wymaga poprawy

## Skrypty

### Eksploracja danych

```bash
.venv/bin/python scripts/01_eda.py
```

Wyświetla:
- Podstawowe statystyki
- Rozkład zmiennych
- Korelacje
- Wartości brakujące

### Trening modeli

```bash
.venv/bin/python scripts/02_train_models.py
```

Proces treningu:
1. Wczytanie i przygotowanie danych
2. Podział train/test (80/20)
3. Trening obu modeli
4. Ewaluacja z metrykami
5. Analiza feature importance
6. Zapis modeli do plików

## Rozwój

### Uruchomienie w trybie deweloperskim

```bash
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testowanie endpointów

Po uruchomieniu serwera, otwórz http://localhost:8000/docs i przetestuj endpointy w interaktywnej dokumentacji Swagger.

## Licencja

MIT

## Autor

Projekt ML - Car Prediction Service
