import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
import time

# ---------------- FLASK SETUP ----------------
app = Flask(__name__)
CORS(app)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_model_advanced.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names_5.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

IMG_SIZE = 224
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print(f"DEBUG: Looking for model at: {MODEL_PATH}")
print(f"DEBUG: Looking for classes at: {CLASS_NAMES_PATH}")

# ---------------- LOAD MODEL & CLASSES ----------------
model = None
class_names = []

def load_real_model():
    global model, class_names
    
    print("🔍 Checking for REAL model files...")
    
    # === LOAD REAL MODEL ===
    if os.path.exists(MODEL_PATH):
        try:
            print("✓ Found REAL model file. Loading...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ REAL MODEL LOADED SUCCESSFULLY!")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape[-1]} classes")
        except Exception as e:
            print(f"✗ ERROR loading model: {e}")
            return False
    else:
        print(f"✗ NO MODEL FILE FOUND at: {MODEL_PATH}")
        print("   Please place your .keras model in model/ folder")
        return False
    
    # === LOAD REAL CLASS NAMES ===
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r") as f:
                class_data = json.load(f)
            
            # Support all formats
            if isinstance(class_data, dict) and "classes" in class_data:
                class_names = class_data["classes"]
            elif isinstance(class_data, list):
                class_names = class_data
            elif isinstance(class_data, dict):
                # {"Class0": 0, "Class1": 1} format
                sorted_items = sorted(class_data.items(), key=lambda x: x[1])
                class_names = [name for name, idx in sorted_items]
            else:
                class_names = [f"Class_{i}" for i in range(model.output_shape[-1])]
            
            print(f"✅ LOADED {len(class_names)} REAL CLASSES: {class_names[:5]}...")
            return True
            
        except Exception as e:
            print(f"✗ ERROR loading classes: {e}")
            # Fallback to numbered classes
            num_classes = int(model.output_shape[-1])
            class_names = [f"SkinDisease_{i}" for i in range(num_classes)]
            print(f"⚠️ Using fallback classes: {num_classes} classes")
            return True
    else:
        print(f"✗ NO CLASS FILE FOUND at: {CLASS_NAMES_PATH}")
        print("   Predictions will use numbered classes (Class_0, Class_1, etc.)")
        num_classes = int(model.output_shape[-1])
        class_names = [f"SkinDisease_{i}" for i in range(num_classes)]
        print(f"⚠️ Using fallback classes: {num_classes} classes")
        return True

# Load at startup
model_loaded = load_real_model()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- HTML UI ----------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Skin Disease Classifier</title>
    <style>
        body { font-family: Arial; max-width: 900px; margin: 50px auto; padding: 20px; }
        .status { padding: 15px; margin: 20px 0; border-radius: 8px; font-weight: bold; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .upload-area { border: 3px dashed #007bff; padding: 60px; text-align: center; margin: 30px 0; border-radius: 10px; }
        .upload-area.dragover { border-color: #0056b3; background: #e7f3ff; }
        button { padding: 12px 30px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .result { background: #e8f5e8; padding: 25px; border-radius: 10px; margin: 20px 0; }
        .prediction { background: #007bff; color: white; padding: 15px; border-radius: 8px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🩺 Skin Disease Classification</h1>
    <div class="status {{ 'success' if model_status else 'warning' }}">{{ status_text }}</div>
    
    <div class="upload-area" id="uploadArea">
        <h3>📁 Upload Skin Image (JPG/PNG)</h3>
        <input type="file" id="imageInput" accept="image/*">
        <br><br>
        <button onclick="uploadImage()">🔬 Analyze Image</button>
    </div>
    
    <div id="resultArea"></div>

    <script>
        async function uploadImage() {
            const file = document.getElementById('imageInput').files[0];
            if (!file) { alert('Please select an image'); return; }
            
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('resultArea').innerHTML = '<div class="status">⏳ Analyzing image...</div>';
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const result = await response.json();
                
                if (result.error) {
                    document.getElementById('resultArea').innerHTML = 
                        `<div class="status error">❌ Error: ${result.error}</div>`;
                } else {
                    let html = `
                        <div class="result">
                            <div class="prediction">
                                <h2>🎯 Top Prediction: <strong>${result.top_prediction}</strong></h2>
                                <p>Confidence: <strong>${result.top_confidence}%</strong></p>
                            </div>
                            <h3>Top 3 Predictions:</h3>
                            <ul style="font-size: 18px;">`;
                    for (let [disease, conf] of Object.entries(result.all_top3)) {
                        html += `<li><strong>${disease}:</strong> ${conf}%</li>`;
                    }
                    html += `</ul></div>`;
                    document.getElementById('resultArea').innerHTML = html;
                }
            } catch (error) {
                document.getElementById('resultArea').innerHTML = 
                    `<div class="status error">❌ Network error: ${error.message}</div>`;
            }
        }
        
        // Drag & drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.ondragover = (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); }
        uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
        uploadArea.ondrop = (e) => {
            e.preventDefault(); uploadArea.classList.remove('dragover');
            document.getElementById('imageInput').files = e.dataTransfer.files;
        };
    </script>
</body>
</html>
"""

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    status_text = "✅ REAL MODEL LOADED" if model_loaded else "⚠️ NO MODEL - Add your .keras file"
    return render_template_string(HTML_TEMPLATE, model_status=model_loaded, status_text=status_text)

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "No model loaded. Place model.keras in model/ folder"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Secure filename
        timestamp = int(time.time())
        filename = f"skin_{timestamp}.jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Predict
        img = preprocess_image(file_path)
        predictions = model.predict(img, verbose=0)[0]

        # Top 3 results
        top_indices = np.argsort(predictions)[-3:][::-1]
        results = {}
        for idx in top_indices:
            if idx < len(class_names):
                class_name = class_names[idx]
                confidence = float(predictions[idx]) * 100
                results[class_name] = round(confidence, 2)

        top_class = list(results.keys())[0]
        top_conf = list(results.values())[0]

        # Cleanup
        os.remove(file_path)

        return jsonify({
            "top_prediction": top_class,
            "top_confidence": top_conf,
            "all_top3": results,
            "success": True
        })

    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "model_loaded": model_loaded,
        "num_classes": len(class_names),
        "class_names": class_names[:5]  # First 5 for preview
    })

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 SKIN DISEASE CLASSIFICATION API - REAL MODEL VERSION")
    print("="*70)
    
    if model_loaded:
        print("✅ SUCCESS: Real model and classes loaded!")
        print(f"📊 {len(class_names)} classes available")
    else:
        print("⚠️  NO MODEL FOUND")
        print("   📁 Create 'model/' folder and add:")
        print("      - skin_disease_model_advanced.keras")
        print("      - class_names.json")
    
    print(f"\n🌐 Open: http://127.0.0.1:5000")
    print("="*70)
    
    app.run(debug=True, host="0.0.0.0", port=5000)
