import os
from flask import Flask, flash, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import cv2
from keras.models import model_from_json
from keras.utils import img_to_array
import numpy as np


app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, ''), exist_ok=True)
app.config['UPLOAD_EXTENSIONS'] = {'.jpg'}

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/image', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return 'Niepoprawny typ pliku'
            f.save(os.path.join(app.instance_path, 'photo', 'image.jpg'))
    return 'Zdjęcie załadowane!'


path = "C:/Users/user/Documents/GitHub/Projects/Emotion-detection/webApp/instance/photo"
dir = os.listdir(path)



if __name__ == "__main__":
    app.run()





