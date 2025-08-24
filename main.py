# FastAPI application for customer churn prediction using ML models
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from utils.inference import predict_new
from utils.config import APP_NAME, VERSION, SECRET_KEY_TOKEN, preprocessor, forest_model, xgboost_model
from utils.CustomerData import CustomerData

# Initialize FastAPI application
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name='X-API-Key')

# Verify API key middleware for endpoint security
async def verify_api_key(api_key: str=Depends(api_key_header)):
    if api_key != SECRET_KEY_TOKEN:
        raise HTTPException(status_code=403, detail="You are not authorized to use this API!")
    return api_key

@app.get('/', tags=['General'])
async def home():
    return {
        "mesage": f"Welcome to {APP_NAME} API v{VERSION}"
    }

# Endpoint for Random Forest model predictions
@app.post('/predict/forest', tags=['Models'])
async def predict_forest(data: CustomerData, api_key: str=Depends(api_key_header)) -> dict:
    try:
        result = predict_new(data=data, preprocessor=preprocessor, model=forest_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for XGBoost model predictions
@app.post('/predict/xgboost', tags=['Models'])
async def predict_xgboost(data: CustomerData, api_key: str=Depends(api_key_header)) -> dict:
    try:
        result = predict_new(data=data, preprocessor=preprocessor, model=xgboost_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))