# backend.py - DermAI Backend (Fixed & Render-ready)
# Run locally: python backend.py
# Deployed on Render: auto-downloads model files from Google Drive

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

app = Flask(__name__, static_folder='.', static_url_path='')

# ----------------------------- CONFIGURATION -----------------------------
# Paths (relative — files will be downloaded to current directory on Render)
WEIGHTS_PATH = "best_weights.weights.h5"
ARCH_PATH = "model_architecture.json"
CLASS_NAMES_PATH = "class_names.json"
SYMPTOMS_PATH = "symptoms.json"
IMG_SIZE = (224, 224)

# Google Drive direct download links (replace YOUR_FILE_ID with real IDs)
WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1f-YTOd67Nw60KEa2IeLXAmpXGGfs7R8O"
ARCH_URL = "https://drive.google.com/uc?export=download&id=1OONpxsXcVyT5caPFmjy_bSzYTJaMRp3w"

# Email Configuration (use environment variables in production/Render!)
EMAIL_SENDER = os.environ.get('EMAIL_SENDER', 'angrajkarn2004@gmail.com')
EMAIL_APP_PASSWORD = os.environ.get('EMAIL_APP_PASSWORD', 'wpjh gfuv ipma ibyi')  # Use App Password only!
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

# ----------------------------- AUTO-DOWNLOAD MODEL FILES -----------------------------
def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {local_path} from {url}...")
        try:
            r = requests.get(url, allow_redirects=True, timeout=60)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded: {local_path}")
        except Exception as e:
            print(f"Download failed for {local_path}: {e}")
            raise RuntimeError(f"Failed to download model file: {local_path}")

# Download files before loading model
print("Checking and downloading model files if missing...")
download_file(WEIGHTS_URL, WEIGHTS_PATH)
download_file(ARCH_URL, ARCH_PATH)

# ----------------------------- LOAD MODEL & DATA -----------------------------
print("Loading model architecture...")
with open(ARCH_PATH, 'r') as f:
    model_json = f.read()
model = tf.keras.models.model_from_json(model_json)

print("Loading weights...")
model.load_weights(WEIGHTS_PATH)

print("Loading class names and symptoms...")
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

with open(SYMPTOMS_PATH, 'r') as f:
    DISEASE_SYMPTOMS = json.load(f)

print(f"Model ready! {len(class_names)} conditions loaded.")

# ----------------------------- HELPER FUNCTIONS -----------------------------
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0
    return img_array

def send_prescription_email(user_info, disease, confidence, match_score, matching, missing, img_bytes):
    msg = MIMEMultipart()
    msg['From'] = f"DermAI Analyzer <{EMAIL_SENDER}>"
    msg['To'] = user_info.get('email', EMAIL_SENDER)
    msg['Subject'] = "Your Skin Condition Analysis Report"

    body = f"""
Dear {user_info.get('name', 'User')},

Thank you for using DermAI Skin Condition Analyzer.

Analysis Summary:
Predicted Condition      : {disease}
Confidence Level         : {confidence}%

Symptom Alignment        : {match_score}
Symptoms You Reported    : {', '.join(user_info.get('symptoms', [])) or 'None provided'}
Matching Symptoms        : {', '.join(matching) or 'None'}
Additional Notes         : {', '.join(missing) or 'All typical symptoms reported'}

General Care Suggestions (for informational purposes):
• Keep the affected area clean and dry.
• Avoid scratching or irritating the skin.
• Use gentle, fragrance-free moisturizers if dryness is present.
• Protect skin from sun exposure.

This report is generated for informational purposes only based on the image and symptoms provided.

Best regards,  
DermAI Support Team
"""

    msg.attach(MIMEText(body, 'plain'))

    # Attach image
    img = MIMEImage(img_bytes)
    img.add_header('Content-Disposition', 'attachment', filename='skin_image.jpg')
    msg.attach(img)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Email sent successfully to {user_info.get('email')}")
    except Exception as e:
        print(f"Email sending failed: {e}")

# ----------------------------- ROUTES -----------------------------
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        img_bytes = file.read()

        # User info
        user_info_str = request.form.get('user_info', '{}')
        user_info = json.loads(user_info_str)

        # Symptoms
        symptoms_str = request.form.get('symptoms', '[]')
        user_symptoms = json.loads(symptoms_str)

        # Predict
        processed = preprocess_image(img_bytes)
        preds = model.predict(processed)[0]
        top_idx = np.argmax(preds)
        confidence = float(preds[top_idx] * 100)
        disease = class_names[top_idx]

        # Symptom match
        known = DISEASE_SYMPTOMS.get(disease, [])
        matching = [s for s in user_symptoms if any(s.lower() == k.lower() for k in known)]
        missing = [k for k in known if not any(k.lower() == u.lower() for u in user_symptoms)]
        match_score = f"{len(matching)} of {len(known)} typical symptoms match" if known else "No symptom data"

        # Send email to the USER'S email
        send_prescription_email(user_info, disease, confidence, match_score, matching, missing, img_bytes)

        return jsonify({
            "disease": disease,
            "confidence": f"{confidence:.2f}",
            "match_score": match_score,
            "matching": matching,
            "missing": missing
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Analysis failed"}), 500

# ----------------------------- RUN SERVER -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render uses env PORT
    print(f"\nDermAI server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

