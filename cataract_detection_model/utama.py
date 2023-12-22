from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
import re

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = 'cataract_detection_model.h5'
loaded_model = load_model(model_path)

# Classes for binary classification
classes = ['Normal', 'Cataract']

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Request Error"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        if file:
            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            file_path = os.path.join("temp", filename)
            file.save(file_path)

            # Preprocess the image
            input_data = preprocess_image(file_path)

            # Make predictions using the loaded model
            predictions = loaded_model.predict(input_data)

            predicted_class = int(np.round(predictions[0][0]))
            percentage = float(predictions[0][0])
            os.remove(file_path)

            result = {
                "percentage": percentage,
                "class": classes[predicted_class]
            }
            return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
