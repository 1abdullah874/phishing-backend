from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and vectorizer
try:
    model = joblib.load("models/phishing_xgb_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
except Exception as e:
    print("Error loading model/vectorizer:", e)
    model, vectorizer = None, None

app = FastAPI(title="Phishing Detection API")

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    email: str

class PredictionResponse(BaseModel):
    is_phishing: bool
    confidence: float
    risk_level: str

@app.get("/")
def root():
    return {"status": "Phishing Detection API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_phishing(request: EmailRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Simple cleaning
        email_text = " ".join(request.email.lower().split())
        features = vectorizer.transform([email_text])

        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        confidence = float(max(probability))
        is_phishing = bool(prediction == 1)

        # Risk level
        if confidence > 0.8:
            risk_level = "high"
        elif confidence > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "is_phishing": is_phishing,
            "confidence": confidence,
            "risk_level": risk_level
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
