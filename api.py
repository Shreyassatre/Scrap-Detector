# api.py

import fastapi
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
from fastapi import File, UploadFile, HTTPException
from huggingface_hub import hf_hub_download
import os

# --- CONFIGURATION ---
# IMPORTANT: Change this to YOUR Hugging Face model repository ID
HF_REPO_ID = "Shreyassatre/Scrap-Detector" 
MODEL_FILENAME = "VGG16_final_model.h5"
CLASSES_FILENAME = "class_names.json"

# Define a cache directory for the model.
# Using a specific directory is good practice for containerized environments.
MODEL_CACHE_DIR = "/tmp/model_cache"

IMG_SIZE = (224, 224)
TOP_N_PREDICTIONS = 4

# --- APP INITIALIZATION ---
app = fastapi.FastAPI(
    title="Scrap Metal Sorter API",
    description="An API that uses a VGG16 model to classify images of scrap metal.",
    version="1.0"
)

# --- GLOBAL VARIABLES ---
# These will be loaded during the startup event
model = None
class_names = []

# --- STARTUP EVENT ---
# This function will run once when the Space starts up.
@app.on_event("startup")
async def load_model_and_classes():
    global model, class_names
    
    print("Application startup: Downloading and loading model...")
    
    # Ensure the cache directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    try:
        # Download model and class names from Hugging Face Hub
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME, cache_dir=MODEL_CACHE_DIR)
        classes_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CLASSES_FILENAME, cache_dir=MODEL_CACHE_DIR)
        
        # Load the model and class names into the global variables
        model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            class_names = json.load(f)
        
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Error during model loading: {e}")

# --- PREDICTION ENDPOINT ---
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not model or not class_names:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load. Check server logs.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMG_SIZE)
        
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    predictions = model.predict(img_array)[0]
    top_n_indices = np.argsort(predictions)[-TOP_N_PREDICTIONS:][::-1]
    
    results = [{"class_name": class_names[i], "confidence": float(predictions[i])} for i in top_n_indices]
    
    return {"filename": file.filename, "predictions": results}

# --- HEALTH CHECK ENDPOINT ---
@app.get("/")
def read_root():
    if model:
        return {"status": "API is running and model is loaded."}
    return {"status": "API is running, but model is NOT loaded. Check logs."}