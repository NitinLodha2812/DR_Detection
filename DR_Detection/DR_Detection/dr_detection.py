from flask import Flask, render_template, url_for, redirect, flash, request
app = Flask(__name__)
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField, FileAllowed

import os
import secrets
from PIL import Image
import cv2
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

import tensorflow as tf
import numpy as np

from models.model import build_model
MODEL_PATH = 'models/trained_model/'
# model = build_model((256,256,3),2)
model = tf.keras.models.load_model(MODEL_PATH)
print('Model loaded.')

from models.gradcam import GradCAM

class UploadForm(FlaskForm):
    picture = FileField('', validators = [FileAllowed(['jpg','png','jpeg'])])
    submit = SubmitField('Upload')

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/pics', picture_fn)

    output_size = (256, 256)
    i = Image.open(form_picture)
    i = i.resize(output_size, Image.ANTIALIAS)
    i.save(picture_path)

    return picture_fn
    
def model_predict(filename, model):
    #Preprocessing the image
    #image = np.array([np.array(Image.open('static/pics/' + filename))]).astype(np.float32)
    #image /= 255.0
    image = cv2.imread('static/pics/'+filename)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    print("shape:", image.shape)
    preds = model.predict(image)
    i = np.argmax(preds[0])
    icam = GradCAM(model, i,None,None)
    heatmap = icam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (256, 256))
    image = np.squeeze(image)
    image *= 255.0
    image = np.asarray(image, np.uint8)
    (heatmap,output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)
    print(heatmap.shape, image.shape)
    cv2.imwrite('static/pics/heatmap_'+filename, heatmap)
    cv2.imwrite('static/pics/heatmap_overlay_'+filename, output)
    return preds
    

@app.route("/", methods=['GET','POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
        return redirect(url_for('prediction', filename = picture_file))
    filename = 'default.png'
    return render_template('home.html', filename = filename, form = form)

@app.route("/prediction")
def prediction():
    filename = request.args['filename']
    # Make prediction on the image
    preds = model_predict(filename, model)

    # Process result to find probability and class of prediction
    pred_prob = "{:.3f}".format(np.amax(preds))    # Max probability
    pred_class = np.argmax(np.squeeze(preds))

    diagnosis = ["No DR", "DR"]

    result = diagnosis[pred_class]               # Convert to string
    return render_template('prediction.html',filename = filename, result = result, prob = pred_prob)

@app.route('/display/<filename>')
def display_image(filename = 'default.png'):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='pics/' + filename), code=301)
    
if __name__ == '__main__':
    app.run(debug=True)
