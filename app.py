import os
import sys
import numpy as np
# FLASK
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# KERAS
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


app = Flask(__name__)
UPLOADS_DIR = 'uploads'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file_path = uploadImage(request.files['image'])
        prediction = modelPredict(file_path, model)
        pred_class = decode_predictions(prediction, top=1) 
        return str(pred_class[0][0][1])
    return None


def uploadImage(file):
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(
        base_path, UPLOADS_DIR, secure_filename(file.filename))
    file.save(file_path)
    return file_path


def modelPredict(file_path, model):
    _image = image.load_img(file_path, target_size=(224, 224))
    arrayImage = image.img_to_array(_image)
    arrayImage = np.expand_dims(_image, axis=0)
    arrayImage = preprocess_input(arrayImage, mode='caffe')
    return model.predict(arrayImage)


PATH_TO_MODEL = 'models/'
model = load_model(PATH_TO_MODEL)
