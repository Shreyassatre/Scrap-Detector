# api.py

import fastapi
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
from fastapi import File, UploadFile, HTTPException
from typing import List

# --- CONFIGURATION ---
MODEL_PATH = 'VGG16_final_model.h5'
CLASSES_PATH = 'class_names.json'
IMG_SIZE = (224, 224)
TOP_N_PREDICTIONS = 4

# --- APP INITIALIZATION ---
app = fastapi.FastAPI(
    title="Scrap Metal Sorter API",
    description="An API that uses a VGG16 model to classify images of scrap metal.",
    version="1.0"
)

# --- MODEL LOADING ---
# Load the model and class names once when the application starts.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    print("✅ Model and class names loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or class names: {e}")
    model = None
    class_names = []


# --- PREDICTION ENDPOINT ---
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, processes it, and returns the top N predictions.
    """
    if not model or not class_names:
        raise HTTPException(status_code=503, detail="Model is not available.")

    # 1. Validate Input File
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # 2. Read and Preprocess the Image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMG_SIZE)
        
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    # 3. Make Prediction
    predictions = model.predict(img_array)[0]
    
    # 4. Format the Output
    top_n_indices = np.argsort(predictions)[-TOP_N_PREDICTIONS:][::-1]
    
    results = [
        {
            "class_name": class_names[i],
            "confidence": float(predictions[i])
        }
        for i in top_n_indices
    ]
    
    return {"filename": file.filename, "predictions": results}

# --- HEALTH CHECK ENDPOINT ---
@app.get("/")
def read_root():
    return {"status": "API is running. Go to /docs for interactive documentation."}