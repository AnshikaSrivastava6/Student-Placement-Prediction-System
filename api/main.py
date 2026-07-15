from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
app=FastAPI()

class StudentData(BaseModel):
    CGPA: float
    Internships: int
    Projects: int
    Workshops_Certifications: int
    AptitudeTestScore: int
    SoftSkillsRating: float
    ExtracurricularActivities: int
    PlacementTraining: int
    SSC_Marks: int
    HSC_Marks: int

@app.get("/")
def home():
    return {
       "message":"Student Placement Prediction System"
    }

model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
@app.post("/predict")
def predict(data:StudentData):
    input_df = pd.DataFrame([data.model_dump()])
    input_df.rename(columns={"Workshops_Certifications": "Workshops/Certifications"},inplace=True)
    input_scaled=scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    not_placed_prob, placed_prob = model.predict_proba(input_scaled)[0]
    return{
        "Prediction": int(prediction),
        "Placed Probability (%)": round(float(placed_prob)*100,3),
        "Not Placed Probability (%)":round(float(not_placed_prob)*100,3)
    }

    