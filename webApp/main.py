import os
from flask import Flask, flash, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
#from app import app

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, ''), exist_ok=True)

#app.config['UPLOAD_PATH'] = 'U'
app.config['UPLOAD_EXTENSIONS'] = {'.png', '.jpg', '.jpeg'}

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#def allowed_file(filename):
#    return '.' in filename and \
 #          filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
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
                abort(400)
            f.save(os.path.join(app.instance_path, 'photo', filename))
    #uploaded_file = request.files['file']
    #filename = secure_filename(uploaded_file.filename)
    #if filename != '':
    #    file_ext = os.path.splitext(filename)[1]
    #    if file_ext not in app.config['UPLOAD_EXTENSIONS']:
    #        abort(400)
    #    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #return redirect(url_for('index'))
    return 'Zdjęcie załadowane!'

if __name__ == "__main__":
    app.run()

