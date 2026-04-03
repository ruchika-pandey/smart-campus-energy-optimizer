# mlops/compare_models.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

model_name = "EnergyPredictor"

# Get all versions
versions = client.search_model_versions(f"name='{model_name}'")
print(f"📊 Found {len(versions)} versions")

best_version = None
best_mae = float('inf')

for v in versions:
    run = client.get_run(v.run_id)
    mae = run.data.metrics.get("mae")
    print(f"Version {v.version}: MAE = {mae:.2f}")
    if mae and mae < best_mae:
        best_mae = mae
        best_version = v.version

if best_version:
    print(f"🏆 Best version: {best_version} (MAE: {best_mae:.2f})")
    
    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=best_version,
        stage="Production"
    )
    print(f"✅ Version {best_version} promoted to Production")