from dotenv import load_dotenv
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from src.predict import load_tf_model, predict_uplift

load_dotenv()

app = FastAPI(
    title="UrbanShift DC API",
    description="Predict uplift potential score for DC neighborhoods.",
    version="0.1.0",
)

# Load the model once at startup
model = load_tf_model()

class UpliftRequest(BaseModel):
    crime_count: float = Field(..., ge=0, description="Total violent+drug crimes for the area.")
    population: float = Field(..., gt=0, description="Population of the area")
    accessibility_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="How accessible the area is (0-1, where 1 is highly accessible).",
    )
    home_value_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Relative home value score (0-1, where 1 is higher value).",
    )

class UpliftResponse(BaseModel):
    uplift_score: float
    crime_rate_per_1000: float

@app.post("/predict", response_model=UpliftResponse)
def predict(request: UpliftRequest):
    try:
        uplift_score, crime_rate_per_1000 = predict_uplift(
            model,
            crime_count=request.crime_count,
            population=request.population,
            accessibility_score=request.accessibility_score,
            home_value_score=request.home_value_score,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return UpliftResponse(
        uplift_score=uplift_score,
        crime_rate_per_1000=crime_rate_per_1000,
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}