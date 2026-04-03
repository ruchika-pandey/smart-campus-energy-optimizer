import mlflow
mlflow.set_tracking_uri("http://localhost:5000")   # ← NEW

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_and_log():
    # Load your data (use your existing data loader)
    df = pd.read_csv('data/raw/research_based_campus_energy.csv')
    
    # Prepare features
    X = df[['hour_of_day', 'temperature', 'occupancy']]
    y = df['energy_kwh']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Start MLflow run
    with mlflow.start_run(run_name="Windows_Training"):
        # Train model
        # Change this line:
        # Change from n_estimators=100 to n_estimators=200
        model = RandomForestRegressor(n_estimators=200, max_depth=15) # was 100, now 200
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("mae", mae)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Run logged! MAE: {mae:.2f}")
        print(f"📊 View at: http://localhost:5000")

if __name__ == "__main__":
    train_and_log()