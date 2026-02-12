"""
Enhanced Model Training Pipeline - Transmission Classification
Trains an improved model for gearbox/transmission classification
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# ============================================================
# SETUP
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_PATH.mkdir(exist_ok=True)

print("=" * 80)
print("ENHANCED TRANSMISSION CLASSIFICATION PIPELINE")
print("=" * 80)
print()

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} records")
print()

# Exploratory analysis for transmission
print("=" * 80)
print("TRANSMISSION ANALYSIS")
print("=" * 80)
transmission_counts = df["Transmission"].value_counts()
print("Transmission Distribution:")
for trans, count in transmission_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {trans}: {count} ({percentage:.1f}%)")
print()

# Cross-tabulation analysis
print("Cross-tabulation Analysis:")
print("\nTransmission by Make:")
print(pd.crosstab(df["Make"], df["Transmission"], normalize="index") * 100)
print("\nTransmission by Fuel Type:")
print(pd.crosstab(df["Fuel Type"], df["Transmission"], normalize="index") * 100)
print()

# ============================================================
# FEATURE ENGINEERING FOR CLASSIFICATION
# ============================================================
print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Create additional features
df_enhanced = df.copy()

# Age feature
df_enhanced["Car_Age"] = 2026 - df_enhanced["Year"]

# Mileage per year
df_enhanced["Mileage_Per_Year"] = df_enhanced["Mileage"] / df_enhanced["Car_Age"]
df_enhanced["Mileage_Per_Year"] = df_enhanced["Mileage_Per_Year"].replace(
    [np.inf, -np.inf], np.nan
)
df_enhanced["Mileage_Per_Year"] = df_enhanced["Mileage_Per_Year"].fillna(
    df_enhanced["Mileage_Per_Year"].median()
)

# Engine size categories
df_enhanced["Engine_Category"] = pd.cut(
    df_enhanced["Engine Size"],
    bins=[0, 1.5, 2.5, 3.5, 5.0],
    labels=["Small", "Medium", "Large", "Very_Large"],
)

# Price categories (quartiles)
price_quartiles = df_enhanced["Price"].quantile([0.25, 0.5, 0.75])
df_enhanced["Price_Category"] = pd.cut(
    df_enhanced["Price"],
    bins=[
        0,
        price_quartiles[0.25],
        price_quartiles[0.5],
        price_quartiles[0.75],
        np.inf,
    ],
    labels=["Low", "Medium", "High", "Premium"],
)

print("Enhanced features created:")
print(
    f"  - Car_Age: {df_enhanced['Car_Age'].min()} to {df_enhanced['Car_Age'].max()} years"
)
print(
    f"  - Mileage_Per_Year: {df_enhanced['Mileage_Per_Year'].min():.0f} to {df_enhanced['Mileage_Per_Year'].max():.0f}"
)
print(f"  - Engine_Category: {df_enhanced['Engine_Category'].value_counts().to_dict()}")
print(f"  - Price_Category: {df_enhanced['Price_Category'].value_counts().to_dict()}")
print()

# Define features for classification
feature_columns = [
    "Make",
    "Model",
    "Year",
    "Engine Size",
    "Mileage",
    "Fuel Type",
    "Price",
    "Car_Age",
    "Mileage_Per_Year",
    "Engine_Category",
    "Price_Category",
]

categorical_features = [
    "Make",
    "Model",
    "Fuel Type",
    "Engine_Category",
    "Price_Category",
]
numerical_features = [
    "Year",
    "Engine Size",
    "Mileage",
    "Price",
    "Car_Age",
    "Mileage_Per_Year",
]

X = df_enhanced[feature_columns]
y_transmission = df_enhanced["Transmission"]

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train_trans, y_test_trans = train_test_split(
    X, y_transmission, test_size=0.2, random_state=42, stratify=y_transmission
)

print(f"Train/Test Split: {len(X_train)} / {len(X_test)}")
print("Train set distribution:")
print(y_train_trans.value_counts(normalize=True) * 100)
print("Test set distribution:")
print(y_test_trans.value_counts(normalize=True) * 100)
print()

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================
print("=" * 80)
print("PREPROCESSING PIPELINE")
print("=" * 80)

# Enhanced preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            categorical_features,
        ),
        ("num", StandardScaler(), numerical_features),
    ]
)

# ============================================================
# MODEL COMPARISON
# ============================================================
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM": SVC(random_state=42, probability=True),
}

best_model = None
best_score = 0
best_model_name = ""
model_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Create pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train_trans, cv=5, scoring="accuracy", n_jobs=-1
    )
    mean_cv_score = cv_scores.mean()
    std_cv_score = cv_scores.std()

    # Train on full training set
    pipeline.fit(X_train, y_train_trans)

    # Test predictions
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test_trans, y_test_pred)
    test_f1 = f1_score(y_test_trans, y_test_pred, average="weighted")

    model_results[name] = {
        "cv_mean": mean_cv_score,
        "cv_std": std_cv_score,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "pipeline": pipeline,
    }

    print(f"  CV Accuracy: {mean_cv_score:.4f} (±{std_cv_score:.4f})")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test F1: {test_f1:.4f}")

    if test_accuracy > best_score:
        best_score = test_accuracy
        best_model = pipeline
        best_model_name = name

print(f"\nBest Model: {best_model_name} (Accuracy: {best_score:.4f})")
print()

# ============================================================
# HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================================
print("=" * 80)
print(f"HYPERPARAMETER TUNING - {best_model_name}")
print("=" * 80)

if best_model_name == "RandomForest":
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
elif best_model_name == "GradientBoosting":
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    }
elif best_model_name == "LogisticRegression":
    param_grid = {
        "model__C": [0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs", "liblinear"],
    }
else:  # SVM
    param_grid = {
        "model__C": [0.1, 1.0, 10.0],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale", "auto"],
    }

print("Starting Grid Search...")
grid_search = GridSearchCV(
    best_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train_trans)

best_pipeline = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print()

# ============================================================
# FINAL MODEL EVALUATION
# ============================================================
print("=" * 80)
print("FINAL MODEL EVALUATION")
print("=" * 80)

# Predictions
y_train_pred = best_pipeline.predict(X_train)
y_test_pred = best_pipeline.predict(X_test)
y_test_proba = best_pipeline.predict_proba(X_test)

# Metrics
train_accuracy = accuracy_score(y_train_trans, y_train_pred)
test_accuracy = accuracy_score(y_test_trans, y_test_pred)
test_precision = precision_score(y_test_trans, y_test_pred, average="weighted")
test_recall = recall_score(y_test_trans, y_test_pred, average="weighted")
test_f1 = f1_score(y_test_trans, y_test_pred, average="weighted")

print("CLASSIFICATION METRICS:")
print(f"  Train Accuracy: {train_accuracy:.4f}")
print(f"  Test Accuracy:  {test_accuracy:.4f}")
print(f"  Test Precision: {test_precision:.4f}")
print(f"  Test Recall:    {test_recall:.4f}")
print(f"  Test F1:        {test_f1:.4f}")
print()

# Overfitting check
if train_accuracy - test_accuracy > 0.1:
    print("Warning: Possible overfitting detected")
else:
    print("No significant overfitting detected")
print()

# Detailed classification report
print("DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test_trans, y_test_pred))
print()

# Confusion matrix
print("CONFUSION MATRIX:")
cm = confusion_matrix(y_test_trans, y_test_pred)
print("                Predicted")
print("                Manual  Automatic")
print(f"Actual Manual     {cm[0, 0]:>6}     {cm[0, 1]:>6}")
print(f"     Automatic    {cm[1, 0]:>6}     {cm[1, 1]:>6}")
print()

# Feature importance (if available)
if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
    print("FEATURE IMPORTANCE:")
    feature_importances = best_pipeline.named_steps["model"].feature_importances_
    feature_names = (
        list(
            best_pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .get_feature_names_out(categorical_features)
        )
        + numerical_features
    )
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    ).sort_values("Importance", ascending=False)

    for i, row in importance_df.head(15).iterrows():
        bar = "█" * int(row["Importance"] * 100)
        print(f"  {row['Feature']:30s} {bar} {row['Importance']:.4f}")
    print()

# ============================================================
# SAVE MODEL
# ============================================================
print("=" * 80)
print("SAVING MODEL")
print("=" * 80)

# Save transmission model
transmission_model_path = MODELS_PATH / "transmission_model.pkl"
joblib.dump(best_pipeline, transmission_model_path)
transmission_size = transmission_model_path.stat().st_size / 1024 / 1024
print(
    f"Transmission model saved: {transmission_model_path} ({transmission_size:.2f} MB)"
)

# Update metadata
metadata_path = MODELS_PATH / "models_metadata.json"
try:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
except FileNotFoundError:
    metadata = {}

metadata["transmission_model"] = {
    "file": "transmission_model.pkl",
    "type": f"{best_model_name}",
    "task": "Classification",
    "target": "Transmission",
    "metrics": {
        "accuracy": float(test_accuracy),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1_score": float(test_f1),
    },
    "features": feature_columns,
    "n_features": len(feature_columns),
    "classes": ["Automatic", "Manual"],
    "best_params": grid_search.best_params_,
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata updated: {metadata_path}")
print()

# ============================================================
# TEST PREDICTIONS
# ============================================================
print("=" * 80)
print("TESTING PREDICTIONS")
print("=" * 80)

# Load model
loaded_model = joblib.load(transmission_model_path)
print("Model loaded successfully")
print()

# Test data
test_data = pd.DataFrame(
    {
        "Make": ["Honda", "BMW", "Ford", "Audi"],
        "Model": ["Model B", "Model A", "Model C", "Model B"],
        "Year": [2020, 2015, 2018, 2022],
        "Engine Size": [2.5, 3.0, 1.8, 2.0],
        "Mileage": [50000, 80000, 45000, 30000],
        "Fuel Type": ["Petrol", "Diesel", "Electric", "Petrol"],
        "Price": [25000, 35000, 28000, 40000],
        "Car_Age": [6, 11, 8, 4],
        "Mileage_Per_Year": [8333, 7273, 5625, 7500],
        "Engine_Category": ["Medium", "Large", "Medium", "Medium"],
        "Price_Category": ["Medium", "High", "High", "Premium"],
    }
)

print("Test Data:")
print(
    test_data[["Make", "Model", "Year", "Engine Size", "Fuel Type"]].to_string(
        index=False
    )
)
print()

# Predictions
transmission_preds = loaded_model.predict(test_data)
transmission_probas = loaded_model.predict_proba(test_data)

print("Transmission Predictions:")
for i, (_, row) in enumerate(test_data.iterrows()):
    pred = transmission_preds[i]
    prob_auto = transmission_probas[i][0]  # Assuming Automatic is first class
    prob_manual = transmission_probas[i][1]
    print(
        f"  {row['Make']} {row['Model']} ({row['Year']}): {pred} "
        f"(Auto: {prob_auto:.2f}, Manual: {prob_manual:.2f})"
    )
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print()

print("TRANSMISSION CLASSIFICATION MODEL:")
print(f"  Algorithm:  {best_model_name}")
print(f"  Accuracy:   {test_accuracy:.4f} ({test_accuracy * 100:.1f}%)")
print(f"  F1-Score:   {test_f1:.4f}")
print("  Status:     Ready for production")
print()

print("SAVED FILES:")
print(f"  - {transmission_model_path.name}")
print("  - models_metadata.json (updated)")
print()

print("NEXT STEPS:")
print("  - Update FastAPI schemas to include transmission prediction")
print("  - Create new endpoint for transmission classification")
print("  - Test API integration")
print()
