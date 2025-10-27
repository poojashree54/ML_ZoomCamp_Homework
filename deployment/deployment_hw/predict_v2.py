import pickle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Load model and DictVectorizer
model_file = 'pipeline_v2.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Initialize FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    customer = await request.json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    result = {'subscription_probability': float(y_pred)}
    return JSONResponse(result)
