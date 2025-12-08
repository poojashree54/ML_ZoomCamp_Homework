import json
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import onnxruntime as ort

def download_image(url: str):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    return Image.open(BytesIO(buffer))

def preprocess(img):
    if img.mode != 'RGB':
        img = img.convert("RGB")
    img = img.resize((200,200), Image.NEAREST)
    X = np.array(img).astype(np.float32)
    X = X / 255.0
    X = X.transpose(2,0,1)
    return np.expand_dims(X,0)

session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name

def lambda_handler(event, context=None):
    url = event.get("url")
    image = download_image(url)
    img = preprocess(image)

    pred = session.run(None, {input_name: img})[0]
    score = float(pred[0][0])

    return { "prediction": score }
