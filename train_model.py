#!/usr/bin/env python3
"""
Yacht Hull Resistance Model Training Script

This script downloads the Delft Yacht Hydrodynamics dataset and trains
multiple machine learning models to predict residuary resistance.
The best performing model is saved for use in the Streamlit application.

Dataset source: UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data
"""

import os
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

# Constants
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
DATA_FILE = "yacht_hydrodynamics.data"
MODEL_DIR = "models"

# Column names for the dataset
COLUMN_NAMES = [
    "longitudinal_position",  # Longitudinal position of center of buoyancy (LC)
    "prismatic_coefficient",  # Prismatic coefficient (PC)
    "length_displacement",    # Length-displacement ratio (L/D)
    "beam_draught",          # Beam-draught ratio (B/Dr)
    "length_beam",           # Length-beam ratio (L/B)
    "froude_number",         # Froude number (Fr)
    "residuary_resistance"   # Residuary resistance per unit weight (Rr) - TARGET
]

FEATURE_NAMES = COLUMN_NAMES[:-1]
TARGET_NAME = COLUMN_NAMES[-1]


def download_dataset(url: str, filename: str) -> str:
    """
    Download the yacht hydrodynamics dataset from UCI repository.

    Args:
        url: URL of the dataset
        filename: Local filename to save the data

    Returns:
        Path to the downloaded file
    """
    if os.path.exists(filename):
        print(f"Dataset already exists at {filename}")
        return filename

    print(f"Downloading dataset from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(filename, 'wb') as f:
        f.write(response.content)

    print(f"Dataset saved to {filename}")
    return filename


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the yacht hydrodynamics dataset.

    The dataset is space-separated with multiple spaces between values.

    Args:
        filepath: Path to the data file

    Returns:
        Preprocessed DataFrame with proper column names
    """
    print("Loading dataset...")

    # Read space-separated data (handles multiple spaces)
    df = pd.read_csv(
        filepath,
        delim_whitespace=True,
        header=None,
        names=COLUMN_NAMES
    )

    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    print(f"\nDataset statistics:\n{df.describe()}")

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for model training.

    Args:
        df: Raw DataFrame

    Returns:
        Tuple of (X, y, feature_names)
    """
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values

    return X, y, FEATURE_NAMES


def train_and_evaluate_models(X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              scaler: StandardScaler) -> dict:
    """
    Train multiple ML models and evaluate their performance.

    Args:
        X_train, X_test: Training and test features (already scaled)
        y_train, y_test: Training and test targets
        scaler: Fitted StandardScaler for feature transformation

    Returns:
        Dictionary containing trained models and their metrics
    """
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )
    }

    results = {}

    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        results[name] = {
            "model": model,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }

        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    return results


def select_best_model(results: dict) -> tuple:
    """
    Select the best performing model based on test R² score.

    Args:
        results: Dictionary of model results

    Returns:
        Tuple of (best_model_name, best_model, best_metrics)
    """
    best_name = max(results.keys(), key=lambda k: results[k]["test_r2"])
    best_result = results[best_name]

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*60}")
    print(f"  Test R²:  {best_result['test_r2']:.4f}")
    print(f"  Test MAE: {best_result['test_mae']:.4f}")
    print(f"  Test RMSE: {best_result['test_rmse']:.4f}")

    return best_name, best_result["model"], best_result


def save_model_artifacts(model, scaler: StandardScaler, model_name: str,
                        metrics: dict, feature_stats: dict, model_dir: str):
    """
    Save the trained model and associated artifacts.

    Args:
        model: Trained model object
        scaler: Fitted StandardScaler
        model_name: Name of the model
        metrics: Performance metrics
        feature_stats: Statistics about features (min, max, mean, std)
        model_dir: Directory to save artifacts
    """
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Save the scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "metrics": {
            "train_r2": float(metrics["train_r2"]),
            "test_r2": float(metrics["test_r2"]),
            "test_mae": float(metrics["test_mae"]),
            "test_rmse": float(metrics["test_rmse"]),
            "cv_mean": float(metrics["cv_mean"]),
            "cv_std": float(metrics["cv_std"])
        },
        "feature_stats": feature_stats
    }

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


def compute_feature_stats(df: pd.DataFrame) -> dict:
    """
    Compute statistics for each feature (for slider ranges in app).

    Args:
        df: DataFrame with all data

    Returns:
        Dictionary with min, max, mean, std for each feature
    """
    stats = {}
    for col in FEATURE_NAMES:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75))
        }

    # Also include target stats
    stats[TARGET_NAME] = {
        "min": float(df[TARGET_NAME].min()),
        "max": float(df[TARGET_NAME].max()),
        "mean": float(df[TARGET_NAME].mean()),
        "std": float(df[TARGET_NAME].std())
    }

    return stats


def save_dataset_for_app(df: pd.DataFrame, model_dir: str):
    """
    Save the processed dataset for use in the Streamlit app.

    Args:
        df: Processed DataFrame
        model_dir: Directory to save the data
    """
    data_path = os.path.join(model_dir, "yacht_data.csv")
    df.to_csv(data_path, index=False)
    print(f"Dataset saved to {data_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("YACHT HULL RESISTANCE PREDICTION - MODEL TRAINING")
    print("="*60)

    # Step 1: Download dataset
    data_file = download_dataset(DATA_URL, DATA_FILE)

    # Step 2: Load and preprocess data
    df = load_and_preprocess_data(data_file)

    # Step 3: Prepare features
    X, y, feature_names = prepare_features(df)

    # Step 4: Split data
    print("\nSplitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Step 5: Scale features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 6: Train and evaluate models
    results = train_and_evaluate_models(
        X_train_scaled, X_test_scaled,
        y_train, y_test, scaler
    )

    # Step 7: Select best model
    best_name, best_model, best_metrics = select_best_model(results)

    # Step 8: Compute feature statistics
    feature_stats = compute_feature_stats(df)

    # Step 9: Save artifacts
    save_model_artifacts(
        best_model, scaler, best_name,
        best_metrics, feature_stats, MODEL_DIR
    )

    # Step 10: Save dataset for app
    save_dataset_for_app(df, MODEL_DIR)

    # Save all model results for comparison in app
    all_results = {
        name: {
            "test_r2": float(r["test_r2"]),
            "test_mae": float(r["test_mae"]),
            "test_rmse": float(r["test_rmse"])
        }
        for name, r in results.items()
    }

    results_path = os.path.join(MODEL_DIR, "all_model_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All model results saved to {results_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nArtifacts saved to '{MODEL_DIR}/' directory.")
    print("Run 'streamlit run streamlit_app.py' to launch the application.")


if __name__ == "__main__":
    main()
