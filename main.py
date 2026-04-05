from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Phishing Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- MODEL LOADING -----------

MODEL_FILE = "models/phishing_xgb_model.pkl"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"

try:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    print("✅ Model and vectorizer loaded successfully")
except Exception as e:
    print("❌ ERROR LOADING MODEL:", e)
    model, vectorizer = None, None

# ----------- REQUEST / RESPONSE -----------

class EmailRequest(BaseModel):
    email: str

class PredictionResponse(BaseModel):
    is_phishing: bool
    confidence: float
    risk_level: str

# ----------- ROUTES -----------

@app.get("/")
def root():
    return {"status": "Phishing Detection API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: EmailRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Clean input
        email_text = " ".join(request.email.lower().split())

        # Transform
        features = vectorizer.transform([email_text])

        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        confidence = float(np.max(probability))
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
