"""
SWE402 Assignment 2 — Task 2d: Model Deployment and Integration

FastAPI service that wraps the trained no-show prediction pipeline
(preprocessing + XGBoost classifier, saved as model/noshow_pipeline.joblib)
and exposes it over HTTP so the n8n agentic workflow (Task 2e) can call it
programmatically.

Run locally:
    pip install fastapi uvicorn pandas scikit-learn xgboost joblib
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger documentation.

Deploy (any of these work, per assignment guidance):
    Render / Railway / Koyeb -> connect this repo, set start command:
        uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from datetime import datetime
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Appointment No-Show Prediction API",
    description="Serves an XGBoost no-show risk model for the SWE402 agentic AI workflow.",
    version="1.0.0",
)

MODEL_PATH = "model/noshow_pipeline.joblib"
model = None


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class PatientAppointment(BaseModel):
    patient_id: str = Field(..., description="Patient identifier, passed through unchanged")
    gender: Literal["M", "F"]
    age: int = Field(..., ge=0, le=120)
    scheduled_day: str = Field(..., description="ISO date the appointment was booked, e.g. 2026-07-01")
    appointment_day: str = Field(..., description="ISO date of the appointment, e.g. 2026-07-15")
    scholarship: Literal[0, 1] = 0
    hypertension: Literal[0, 1] = 0
    diabetes: Literal[0, 1] = 0
    alcoholism: Literal[0, 1] = 0
    handicap: Literal[0, 1] = 0
    sms_received: Literal[0, 1] = 0


class PredictionResponse(BaseModel):
    patient_id: str
    noshow_probability: float
    noshow_prediction: Literal["Yes", "No"]
    risk_level: Literal["Low", "Medium", "High"]
    recommended_action: str


class BatchRequest(BaseModel):
    appointments: list[PatientAppointment]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def build_features(appt: PatientAppointment) -> pd.DataFrame:
    sched = datetime.fromisoformat(appt.scheduled_day)
    apt = datetime.fromisoformat(appt.appointment_day)
    waiting_days = (apt.date() - sched.date()).days
    if waiting_days < 0:
        raise HTTPException(status_code=422, detail="appointment_day cannot be before scheduled_day")

    row = {
        "Gender": appt.gender,
        "Age": appt.age,
        "Scholarship": appt.scholarship,
        "Hypertension": appt.hypertension,
        "Diabetes": appt.diabetes,
        "Alcoholism": appt.alcoholism,
        "Handicap": appt.handicap,
        "SMS_received": appt.sms_received,
        "WaitingDays": waiting_days,
        "AppointmentDayOfWeek": apt.strftime("%A"),
    }
    return pd.DataFrame([row])


def risk_band(prob: float) -> tuple[str, str]:
    if prob >= 0.55:
        return "High", "Call patient directly to confirm attendance and offer rescheduling."
    elif prob >= 0.30:
        return "Medium", "Send a personalised SMS reminder 24h before the appointment."
    else:
        return "Low", "Standard automated reminder; no additional action needed."


def predict_one(appt: PatientAppointment) -> PredictionResponse:
    X = build_features(appt)
    prob = float(model.predict_proba(X)[0, 1])
    pred = "Yes" if prob >= 0.5 else "No"
    level, action = risk_band(prob)
    return PredictionResponse(
        patient_id=appt.patient_id,
        noshow_probability=round(prob, 4),
        noshow_prediction=pred,
        risk_level=level,
        recommended_action=action,
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(appt: PatientAppointment):
    return predict_one(appt)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(predictions=[predict_one(a) for a in req.appointments])
