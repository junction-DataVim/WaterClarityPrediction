from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import json

# Load artifacts
model = joblib.load('water_quality_model.pkl')
with open('feature_names.json', 'r') as f:
    FEATURE_NAMES = json.load(f)
with open('class_labels.json', 'r') as f:
    CLASS_LABELS = json.load(f)

app = FastAPI(
    title="Water Quality Prediction API",
    description="API for predicting water quality class based on 7 parameters",
    version="1.0.0"
)

class WaterQualityInput(BaseModel):
    temp: float
    turbidity: float
    do: float
    bod: float
    ph: float
    ammonia: float
    nitrite: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "temp": 67.45,
                "turbidity": 10.13,
                "do": 0.208,
                "bod": 7.474,
                "ph": 4.752,
                "ammonia": 0.286,
                "nitrite": 4.355
            }]
        }
    }

@app.post("/predict", summary="Predict water quality class")
async def predict(input_data: WaterQualityInput):
    """
    Predict water quality class (0=Excellent, 1=Good, 2=Poor) based on:
    
    - **temp**: Temperature (°C)
    - **turbidity**: Turbidity (cm)
    - **do**: Dissolved Oxygen (mg/L)
    - **bod**: Biological Oxygen Demand (mg/L)
    - **ph**: pH value
    - **ammonia**: Ammonia concentration (mg/L)
    - **nitrite**: Nitrite concentration (mg/L)
    """
    # Convert input to array in correct feature order
    input_array = np.array([
        input_data.temp,
        input_data.turbidity,
        input_data.do,
        input_data.bod,
        input_data.ph,
        input_data.ammonia,
        input_data.nitrite
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_array)[0]
        prob_dict = {CLASS_LABELS[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
    else:
        prob_dict = None
        
    return {
        "quality": CLASS_LABELS[str(prediction)],
        "probabilities": prob_dict
    }

@app.get("/")
async def health_check():
    return {
        "status": "active",
        "model": type(model.named_steps['classifier']).__name__ if hasattr(model, 'named_steps') else type(model).__name__,
        "features": FEATURE_NAMES,
        "classes": CLASS_LABELS
    }