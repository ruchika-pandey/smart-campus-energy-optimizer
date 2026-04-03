# api/app.py
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(title="Smart Campus Energy API")

# Try to load model, use fallback if not found
model = None
try:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    model = mlflow.pyfunc.load_model("models:/EnergyPredictor1/Production")
    print("✅ Production model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load production model: {e}")
    print("⚠️ Using fallback simulation")

class PredictionRequest(BaseModel):
    building: str
    hour: int
    temperature: float
    occupancy: float
    weekend: bool = False
    event: bool = False

class PredictionResponse(BaseModel):
    predicted_energy: float
    optimized_energy: float
    savings_kwh: float
    savings_cost: float
    co2_reduction: float

@app.get("/")
def root():
    return {"message": "API is running", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        if model is not None:
            features = pd.DataFrame([{
                'hour_of_day': req.hour,
                'temperature': req.temperature,
                'occupancy': req.occupancy,
            }])
            pred = model.predict(features)[0]
            
            building_factors = {
                'Library': 1.2, 'Computer Lab': 1.5, 'Classroom': 1.0,
                'Hostel': 0.8, 'Auditorium': 1.8
            }
            pred *= building_factors.get(req.building, 1.0)
        else:
            # Fallback simulation
            pred = 100 + req.hour * 8
            building_factors = {
                'Library': 1.2, 'Computer Lab': 1.5, 'Classroom': 1.0,
                'Hostel': 0.8, 'Auditorium': 1.8
            }
            pred *= building_factors.get(req.building, 1.0)
            if req.temperature > 28:
                pred *= 1.3
            if req.occupancy > 0.7:
                pred *= 1.2
            if req.event:
                pred *= 1.4
            if req.weekend:
                pred *= 0.7

        optimized = pred * 0.85
        saved = pred - optimized
        cost = saved * 8
        co2 = saved * 0.82

        return PredictionResponse(
            predicted_energy=round(pred, 2),
            optimized_energy=round(optimized, 2),
            savings_kwh=round(saved, 2),
            savings_cost=round(cost, 2),
            co2_reduction=round(co2, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)