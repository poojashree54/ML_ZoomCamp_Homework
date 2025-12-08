FROM agrigorev/model-2025-hairstyle:v1

RUN pip install pillow numpy onnxruntime

COPY lambda_handler.py .

CMD ["lambda_handler.lambda_handler"]
