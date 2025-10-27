import pickle
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

model_file = 'pipeline_v1.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    customer = await request.json()

    X = dv.transform([customer])
    probability = model.predict_proba(X)[0, 1]

    result = {"subscription_probability": probability}

    return JSONResponse(result)

if __name__ == "__main__":
    uvicorn.run("predict_v1:app", host="0.0.0.0", port=9696, reload=True)