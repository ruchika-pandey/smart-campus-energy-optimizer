# test_load_model.py
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
model_name = "EnergyPredictor"

# Load the Production model
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")
print("✅ Production model loaded successfully")

# Test prediction with sample data
sample = pd.DataFrame({
    'hour_of_day': [14],
    'temperature': [32],
    'occupancy': [0.8]
})
pred = model.predict(sample)
print(f"🔮 Sample prediction: {pred[0]:.2f} kWh")