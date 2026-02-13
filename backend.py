# backend.py - Flask Backend for DermAI Frontend
# Run this file: python backend.py
# Then open http://127.0.0.1:5000 in your browser

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os

app = Flask(__name__, static_folder='.', static_url_path='')

# ----------------------------- CONFIGURATION -----------------------------
# Use raw strings for Windows paths (prevents unicode escape errors)
WEIGHTS_PATH = r"C:\Users\admin\Desktop\skin-disease-app\best_weights.weights.h5"
ARCH_PATH = r"C:\Users\admin\Desktop\skin-disease-app\model_architecture.json"
CLASS_NAMES_PATH = r"C:\Users\admin\Desktop\skin-disease-app\class_names.json"
IMG_SIZE = (224, 224)
PORT = 5000

# Load model architecture from JSON
print("Loading model architecture from JSON...")
with open(ARCH_PATH, 'r') as f:
    model_json = f.read()
model = tf.keras.models.model_from_json(model_json)

# Load trained weights
print("Loading trained weights...")
model.load_weights(WEIGHTS_PATH)

# Load class names (must be a list in JSON)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

if not isinstance(class_names, list):
    raise ValueError("class_names.json must be a list of class names")

print(f"Model fully loaded! {len(class_names)} classes ready.")

# ----------------------------- PREPROCESS FUNCTION -----------------------------
def preprocess_image(img_bytes):
    """Load image from bytes, resize, and preprocess for EfficientNet"""
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')
    
    img = img.resize(IMG_SIZE)
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0
    
    return img_array

# ----------------------------- ROUTES -----------------------------
@app.route('/')
def serve_index():
    """Serve the index.html frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive uploaded image, predict, and return top-1 result"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400
        
        img_bytes = file.read()
        processed_img = preprocess_image(img_bytes)
        
        predictions = model.predict(processed_img)[0]
        
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx] * 100)
        disease = class_names[top_idx]
        
        return jsonify({
            "disease": disease,
            "confidence": f"{confidence:.2f}"
        })
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Prediction failed"}), 500

# ----------------------------- RUN SERVER -----------------------------
if __name__ == '__main__':
    print(f"\n🚀 DermAI server starting on http://127.0.0.1:{PORT}")
    print("Open the link in your browser to use the app.\n")
    app.run(host='127.0.0.1', port=PORT, debug=False)