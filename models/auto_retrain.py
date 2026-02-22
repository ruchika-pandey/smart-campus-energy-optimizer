# models/auto_retrain.py
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use local file tracking (no server needed)
mlflow.set_tracking_uri("file:./mlruns")

def load_training_data():
    """Try to load real data, fallback to synthetic."""
    data_path = "data/raw/research_based_campus_energy.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"✅ Loaded real data: {len(df)} rows")
    else:
        print("⚠️ Real data not found, generating synthetic data...")
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'hour_of_day': np.random.randint(0, 24, n),
            'temperature': np.random.uniform(15, 35, n),
            'occupancy': np.random.uniform(0.1, 0.9, n),
            'energy_kwh': np.random.uniform(50, 300, n)
        })
        # Make synthetic energy realistic
        df['energy_kwh'] = (
            100 + 20 * np.sin(2 * np.pi * df['hour_of_day'] / 24)
            + (df['temperature'] - 22) * 3
            + df['occupancy'] * 50
            + np.random.normal(0, 10, n)
        )
    return df

def train_and_log():
    df = load_training_data()
    feature_cols = ['hour_of_day', 'temperature', 'occupancy']
    X = df[feature_cols]
    y = df['energy_kwh']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    with mlflow.start_run(run_name="auto_retrain_github"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 12)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id
        print(f"✅ Run logged (ID: {run_id}) with MAE = {mae:.2f}")

if __name__ == "__main__":
    train_and_log()