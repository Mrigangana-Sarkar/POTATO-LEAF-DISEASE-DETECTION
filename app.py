import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Suppress oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Ensure the static folder exists
os.makedirs('static', exist_ok=True)

# Load the model
model = tf.keras.models.load_model('model.h5', compile=False)
class_names = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']
IMAGE_SIZE = 255

# Prediction function
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.image.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize and batch

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            predicted_class, confidence = predict(img)

            return render_template('index.html', 
                                   image_path=filepath, 
                                   predicted_label=predicted_class, 
                                   confidence=confidence)
    return render_template('index.html', message='Upload an image')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)