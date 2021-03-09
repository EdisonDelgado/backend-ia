import os

import numpy
import numpy as np
# FLASK
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
# KERAS
from keras.models import load_model
from keras.preprocessing import image

# from flask_cors import CORS, cross_origin

app = Flask(__name__)
# app.config['CORS_HEADERS'] = 'Content-Type'
port = int(os.environ.get("PORT", 5000))
UPLOADS_DIR = 'uploads'
training = ['Bacterial_spot',
            'Early_blight',
            'Late_blight',
            'Leaf_Mold',
            'Septoria_leaf_spot',
            'Spider_mites Two-spotted_spider_mite',
            'Target_Spot',
            'Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato_mosaic_virus',
            'Healthy']


@app.route('/predict', methods=['POST'])
# @cross_origin()
def predict():
    if request.method == 'POST':
        file_path = uploadImage(request.files['image'])
        prediction = modelPredict(file_path)
        indexes = numpy.argmax(prediction, axis=1)
        return jsonify(status=True, prediction=str(training[indexes[0]]))
    return None


def uploadImage(file):
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(
        base_path, UPLOADS_DIR, secure_filename(file.filename))
    file.save(file_path)
    return file_path


def modelPredict(file_path):
    base_path = os.path.dirname(__file__)
    model = load_model(os.path.abspath(os.path.join(base_path, "modelV2.h5")))
    _image = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(_image)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
