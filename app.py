# [Author  : ] -> Ameer Hamza
# [Release : ] -> 21.12.13
# [Permission Protocol : ] -> 21X-Python-UBA

from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import pandas as pd

# tensorflow & Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
import tensorflow
from tensorflow import keras
from keras.models import Sequential,load_model,Model
from keras.layers import MaxPool2D,Dense,Input,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.applications import resnet

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model Building
print("****** Building Custom Resnet Model *******")

base_model_tf=resnet.ResNet50(include_top=False,input_shape=(224,224,3),classes=38)
base_model_tf.trainable=False

pt=Input(shape=(224,224,3))
func=tensorflow.cast(pt,tensorflow.float32)
x=preprocess_input(func) 
model=base_model_tf(x,training=False)
model=GlobalAveragePooling2D()(model)
model=Dense(128,activation='relu')(model)
model=Dense(64,activation='relu')(model)
model=Dense(38,activation='softmax')(model)
model=Model(inputs=pt,outputs=model)

# print("***************** Model Summary ********************")
# model.summary()
# print("****************************************************")

# Load your trained model
print("****** Loading Model Weights *******")
MODEL_PATH = 'models/MobileNetV2.h5'
model = load_model(MODEL_PATH)

print("****** Loading CSV File ********")
data = pd.read_csv("types of diseases.csv")

print("****** Model Loaded *******")
print('Check http://127.0.0.1:5000/')


classes_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 
    'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 
    'Raspberry___healthy', 
    'Soybean___healthy', 
    'Squash___Powdery_mildew', 
    'Strawberry___healthy', 'Strawberry___Leaf_scorch', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']


def model_predict(img_path, model):
    
    img = tensorflow.keras.utils.load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        pred_class = preds.argmax(axis=-1)           
        print(pred_class)
        idx = pred_class[0]
        ['classes_name', 'disease_name', 'remedy']
        datax = "\
                <div>\
                <h5>Results:</h5>\
                <p style='font-size: 20px'>"+data['disease_name'][idx]+"</p>\
                <p style='font-size: 12px'>"+data['classes_name'][idx]+"</p>\
                <h5>Remedies:</h5>\
                <p style='font-size: 20px'>"+data['remedy'][idx]+"</p>\
                </div>\
                "
        return datax
    
    return None


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", debug=True)

