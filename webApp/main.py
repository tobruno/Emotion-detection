import os
from flask import Flask, flash, render_template, request
from werkzeug.utils import secure_filename
import cv2
from keras.models import model_from_json
from keras.utils import img_to_array
import numpy as np


app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, ''), exist_ok=True)
app.config['UPLOAD_EXTENSIONS'] = {'.jpg', '.JPG'}

def detect():
    model = model_from_json(open("C:/Users/user/Documents/GitHub/Projects/Emotion-detection/Model1/model1.json", "r").read())
    image = cv2.imread('C:/Users/user/Documents/GitHub/Projects/Emotion-detection/webApp/instance/photo/image.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (48, 48))
    image_pixels = img_to_array(gray_image)
    image_pixels = np.expand_dims(image_pixels, axis=0)

    predictions = model.predict(image_pixels)
    max_index = np.argmax(predictions[0])

    emotion_detection = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral')
    emotion_prediction = emotion_detection[max_index]
    print(emotion_prediction)
    return emotion_prediction


path = "C:/Users/user/Documents/GitHub/Projects/Emotion-detection/webApp/instance/photo"
dir = os.listdir(path)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/image', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return 'Niepoprawny typ pliku (wymagany .jpg)'
            f.save(os.path.join(app.static_folder, 'image.jpg'))

    return render_template("image.html")


@app.route('/result')
def result():
    emotion = detect()
    return render_template("result.html") + emotion



if __name__ == "__main__":
    app.run()
    print(len(dir))
