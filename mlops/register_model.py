# mlops/register_model.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

# Replace with your actual run_id from the last training
run_id = "5451233058c441a4b72ad8314dc18635"  # e.g., "a1b2c3..."

model_name = "EnergyPredictor"

# Register the model
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

print(f"✅ Registered model: {result.name} version {result.version}")

# Optionally add a description
client.update_model_version(
    name=model_name,
    version=result.version,
    description="Random Forest model with 100 estimators"
)