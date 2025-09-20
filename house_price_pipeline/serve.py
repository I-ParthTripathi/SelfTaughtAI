from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('house_pipeline.joblib')

@app.post('/predict')
def predict(payload: dict):
    df = pd.DataFrame([payload])
    preds = model.predict(df)
    return {'pred': float(preds[0])}