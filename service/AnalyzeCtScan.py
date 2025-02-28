import os
import cv2
import numpy as np
from flask import jsonify
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'uploaded_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def analyzeCtScan(request):
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            model_path = os.path.join(os.path.dirname(__file__), "../models/modelbrainTumor.h5")
            print(model_path)
            model = load_model(model_path)

            diagnosis , abnormality_score = predict_tumor(file_path, model)

            # Delete the files after sending the response
            os.remove(file_path)

            response_data = {
                "diagnosis": diagnosis,
                "abnormality_score": abnormality_score
            }

            return jsonify(response_data)

        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({"error": "Error processing file"}), 500


def img_process(file):
  img = cv2.imread(file)
  img = cv2.resize(img, (256,256))
  img = img/255
  return img


def predict_tumor(file_path, model):
    """Predicts the tumor class and confidence level for a given image"""
    category = {
        0: "pituitary",
        1: "notumor",
        2: "meningioma",
        3: "glioma"
    }

    # Preprocess image
    img = img_process(file_path)

    # Get prediction
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    p_class = category[predicted_class[0]]
    p_confidence = prediction[0][predicted_class[0]] * 100

    return p_class , p_confidence
