from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import re # Regular expressions for text pattern matching

app = FastAPI()

# Load your saved models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
selector = joblib.load(os.path.join(BASE_DIR, 'phishing_rfe_selector.pkl'))
model = joblib.load(os.path.join(BASE_DIR, 'final_phishing_model.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'target_label_encoder.pkl'))

# 1. YOUR CUSTOM FEATURE EXTRACTION FUNCTION
def extract_32_features(email_text: str):
    """
    This function analyzes the raw email text and outputs 
    a list of 32 values consisting of -1, 0, or 1.
    """
    features = []
    
    # Example Feature 1: Check character length
    if len(email_text) < 50:
        features.append(-1) # Safe/Normal
    elif len(email_text) < 200:
        features.append(0)  # Suspicious
    else:
        features.append(1)  # High risk
        
    # Example Feature 2: Check for urgent keywords
    urgent_words = ['urgent', 'action required', 'verify', 'suspend', 'login']
    count = sum(1 for word in urgent_words if word in email_text.lower())
    if count == 0:
        features.append(-1)
    elif count <= 2:
        features.append(0)
    else:
        features.append(1)
        
    # ... Write your logic here to append the remaining 30 features ...
    # For testing, let's fill the rest with dummy values (0) until you write all 32
    while len(features) < 32:
        features.append(0)
        
    return features

# Define the structure for incoming API requests (Now accepting raw text!)
class EmailPayload(BaseModel):
    email_content: str

@app.post("/predict")
def predict_email(data: EmailPayload):
    # 2. Transform the raw text into your 32 numerical attributes
    numerical_features = extract_32_features(data.email_content)
    
    # 3. Convert to DataFrame for the scikit-learn models
    input_df = pd.DataFrame([numerical_features])
    
    # 4. Filter with RFE and make the prediction
    selected_data = selector.transform(input_df)
    numeric_prediction = model.predict(selected_data)[0]
    string_label = label_encoder.inverse_transform([numeric_prediction])[0]
    
    return {
        "extracted_features_preview": numerical_features[:5], # Show first 5 scores
        "prediction": int(numeric_prediction),
        "label": str(string_label)
    }