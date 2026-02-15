
 Problem Statement

Skin diseases are among the most common health issues worldwide, affecting millions of people every year. Conditions like acne, eczema, psoriasis, fungal infections, and even skin cancer can significantly impact quality of life. Early detection is crucial, especially for serious conditions like melanoma, but access to dermatologists is limited — particularly in rural areas or developing regions like India, where there may be only one dermatologist for thousands of patients.

DermAI is an educational AI project that demonstrates how deep learning can help identify 23 common skin conditions from a simple photo. It is **not a medical diagnostic tool** — it is built for learning purposes only and should never replace professional medical advice.

 Project Overview

DermAI is a web-based application that allows users to upload a skin image and get an AI-generated prediction of possible skin conditions along with confidence scores. It uses a Convolutional Neural Network (CNN) based on EfficientNetB0, trained on the public DermNet dataset.

Important Disclaimer: This is a student/academic project. Predictions may be inaccurate. Always consult a qualified dermatologist for any skin concerns.

 File Structure


skin-disease-app/
├── backend.py                      Flask backend server (main script)
├── index.html                    
  Main frontend page (upload & results)
├── about.html                      About the model page (technical details)
├── class_names.json                
List of 23 class names (in order)
├── best_weights.weights.h5        
 Trained model weights
├── model_architecture.json      
  Model structure (JSON format)
└── README.md                      

Requirements

- Python 3.8 or higher
- The following Python packages:
  - Flask
  - TensorFlow (CPU version is fine; works without GPU)
  - Pillow (PIL)
  - NumPy

 Installation

1. Clone or download this project folder.
2. Open a terminal/command prompt in the project folder.
3. Create a virtual environment (optional but recommended):
   
   python-m venv venv
   venv\Scripts\activate    # On Windows
   # source venv/bin/activate  # On macOS/Linux
   
4. Install dependencies:
   
   pip install flask tensorflow pillow numpy
   

**Note**: TensorFlow may take a few minutes to install and can be large (~500 MB). A CPU-only version is used by default.

 Setup & How to Run

1. Ensure all files listed in the File Structure are in the same folder.
2. Open a terminal in the project folder.
3. Run the backend server:
   
   python backend.py
  
   You should see:
   
   Model fully loaded! 23 classes ready.
   🚀 DermAI server starting on http://127.0.0.1:5000
   

4. Open your web browser and go to:  
   http://127.0.0.1:5000

5. On the homepage:
   - Click "Upload & Analyze" or the upload box.
   - Select a clear skin photo (JPG/PNG).
   - The AI will analyze and show the predicted condition with confidence.

6. Click "About Model" in the navigation bar for detailed technical information.

To stop the server, press `Ctrl + C` in the terminal.

Model Details (Brief)

- Base Model  EfficientNetB0 (pre-trained on ImageNet)
- Dataset: DermNet (~19,000 images, 23 classes)
- Accuracy: ~41% top-1, ~75% top-5 on test data
- Input: 224x224 RGB images
- Preprocessing: Scaled to [-1, 1] range (same as training)

## Troubleshooting

- If you get a 404 error when opening the link: Make sure `index.html` is in the same folder as `backend.py`.
- Model loading errors: Ensure the weights (`.weights.h5`) and architecture (`.json`) files are present and not corrupted.
- Slow predictions: Running on CPU — normal for first-time use (TensorFlow loads slowly initially).



 