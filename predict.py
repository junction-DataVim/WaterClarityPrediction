import joblib
import json
import numpy as np
import pandas as pd

# Load artifacts
model = joblib.load('water_quality_model.pkl')
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

def predict_water_quality(temp, turbidity, do, bod, ph, ammonia, nitrite):
    """
    Predict water quality from input parameters
    
    Parameters:
    temp: Temperature (°C)
    turbidity: Turbidity (cm)
    do: Dissolved Oxygen (mg/L)
    bod: Biological Oxygen Demand (mg/L)
    ph: pH value
    ammonia: Ammonia concentration (mg/L)
    nitrite: Nitrite concentration (mg/L)
    
    Returns: dict with prediction details
    """
    # Create input DataFrame with proper feature names
    input_data = pd.DataFrame([[temp, turbidity, do, bod, ph, ammonia, nitrite]], 
                             columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    quality = class_labels[str(prediction)]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)[0]
        prob_dict = {class_labels[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
    else:
        prob_dict = None
        
    return {
        'prediction': int(prediction),
        'quality': quality,
        'probabilities': prob_dict
    }

# Example usage
if __name__ == "__main__":
    sample_prediction = predict_water_quality(
        temp=67.45,
        turbidity=10.13,
        do=0.208,
        bod=7.474,
        ph=4.752,
        ammonia=0.286,
        nitrite=4.355
    )
    print("Sample prediction:", sample_prediction)