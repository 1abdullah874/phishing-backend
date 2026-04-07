PhishGuard – AI-Powered Phishing Detection Web App
<p align="center">
  <img src="https://img.shields.io/badge/Status-Live-success" alt="Status">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/XGBoost-EB6C34?logo=xgboost&logoColor=white" alt="XGBoost">
</p>
<p align="center">
  <strong>🔗 <a href="https://phishing-shield.vercel.app/">Live Demo</a></strong>
</p>

📋 Overview
PhishGuard is a full-stack web application that detects whether an email is phishing or safe using machine learning. Users can paste email content into the app and receive instant analysis with confidence scores and risk levels.

✨ Features

🎯 Real-time phishing detection – Instant analysis of email content
🎨 Clean and simple web interface – User-friendly design
🚀 REST API backend – Scalable and efficient
🤖 Machine learning-based predictions – Powered by XGBoost
☁️ Fully deployed – Accessible anywhere, anytime


🛠️ Tech Stack
Backend

FastAPI – Modern, fast web framework
Uvicorn – Lightning-fast ASGI server
Scikit-learn – TF-IDF Vectorizer for feature extraction
XGBoost – High-performance machine learning algorithm
Joblib – Model serialization

Frontend

React – Interactive UI components
Vite – Next-generation frontend tooling

Deployment

Railway – Backend hosting
Vercel – Frontend hosting


🔄 How It Works
1. User inputs email text
   ↓
2. Text is cleaned and processed
   ↓
3. TF-IDF vectorizer converts text into numerical features
   ↓
4. XGBoost model predicts phishing probability
   ↓
5. API returns structured response with risk assessment

🌐 API Endpoints
GET /health
Check if the API is running and the model is loaded
Response:
json{
  "status": "healthy",
  "model_loaded": true
}
POST /predict
Analyze email content for phishing detection
Request:
json{
  "email": "Your email text here"
}
Response:
json{
  "is_phishing": true,
  "confidence": 0.92,
  "risk_level": "dangerous"
}
Risk Levels:

🟢 Safe – Email appears legitimate
🟡 Suspicious – Potential phishing indicators detected
🔴 Dangerous – High probability of phishing


🤖 Model Details

Algorithm: XGBoost Classifier
Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency)
Input: Raw email text
Output: Binary classification (Phishing / Safe) with confidence score


⚠️ Limitations

The model is trained on a relatively small dataset, so accuracy may vary
May misclassify complex or novel phishing patterns
Performance can be improved with:

Larger and more diverse datasets
Advanced feature engineering
Deep learning NLP models (BERT, GPT)




🚀 Future Improvements

 Integrate larger and more diverse phishing datasets
 Add explainability features (highlight suspicious phrases)
 URL and domain reputation analysis
 User authentication and detection history tracking
 Chrome extension for real-time email scanning
 Multi-language support
 Advanced NLP models (transformer-based)


📊 Project Status
✅ Fully working full-stack application
✅ Deployed and accessible online
🔄 Actively being improved

👤 Author
Abdullah Ali Saleem

🙏 Acknowledgment
This project was built as a learning experience to understand:

Machine learning pipelines and deployment
Backend API development with FastAPI
Frontend integration with React
Real-world application deployment
End-to-end software engineering
