import mlflow
from mlflow.tracking import MlflowClient

# Connect to your local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Set version 1 of "EnergyPredictor1" to Production
client.transition_model_version_stage(
    name="EnergyPredictor1",
    version=1,
    stage="Production"
)
print("✅ Version 1 of EnergyPredictor1 is now in Production.")