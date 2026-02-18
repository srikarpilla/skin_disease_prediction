# backend.py - DermAI Backend with Formal Email Prescription Report
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

app = Flask(__name__, static_folder='.', static_url_path='')

# ----------------------------- CONFIGURATION -----------------------------
WEIGHTS_PATH = r"C:\Users\Srikar\Downloads\skin_disease_prediction-main\best_weights.weights.h5"
ARCH_PATH = r"C:\Users\Srikar\Downloads\skin_disease_prediction-main\model_architecture.json"
CLASS_NAMES_PATH = r"C:\Users\Srikar\Downloads\skin_disease_prediction-main\class_names.json"
SYMPTOMS_PATH = r"C:\Users\Srikar\Downloads\skin_disease_prediction-main\symptoms.json"
IMG_SIZE = (224, 224)
IMG_SIZE = (224, 224)
PORT = 5000

# Email Configuration - CHANGE THESE
EMAIL_SENDER = 'angrajkarn2004@gmail.com'          # Your Gmail
EMAIL_APP_PASSWORD = 'wpjh gfuv ipma ibyi'  # Gmail App Password
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

# Load model & data (same as before)
print("Loading model...")
with open(ARCH_PATH, 'r') as f:
    model_json = f.read()
model = tf.keras.models.model_from_json(model_json)
model.load_weights(WEIGHTS_PATH)

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

with open(SYMPTOMS_PATH, 'r') as f:
    DISEASE_SYMPTOMS = json.load(f)

print(f"Ready – {len(class_names)} conditions loaded.")

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
    msg['To'] = user_info['email']
    msg['Subject'] = "Your Skin Condition Analysis Report"

    body = f"""
Dear {user_info['name']},

Thank you for using DermAI Skin Condition Analyzer.

Analysis Summary:
Predicted Condition      : {disease}
Confidence Level         : {confidence}%

Symptom Alignment        : {match_score}
Symptoms You Reported    : {', '.join(user_info['symptoms'].split(',')) if user_info['symptoms'] else 'None provided'}
Matching Symptoms        : {', '.join(matching) if matching else 'None'}
Additional Notes         : {', '.join(missing) if missing else 'All typical symptoms reported'}

General Care Suggestions (for informational purposes):
• Keep the affected area clean and dry.
• Avoid scratching or irritating the skin.
• Use gentle, fragrance-free moisturizers if dryness is present.
• Protect skin from sun exposure.

This report is generated for informational purposes only based on the image and symptoms provided.
It is not a substitute for professional medical evaluation.

Please consult a qualified dermatologist or healthcare provider for a proper diagnosis and treatment plan.

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
        print(f"Prescription email sent to {user_info['email']}")
    except Exception as e:
        print(f"Email failed: {e}")

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

        # Send email
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

if __name__ == '__main__':
    print("\nDermAI server running at http://127.0.0.1:5000\n")
    app.run(host='127.0.0.1', port=PORT, debug=False)
