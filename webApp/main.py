from flask import Flask
app = Flask(__name__)

@app.route('/')
def emodetect():
    return '<head> Emotion detection! <h1> SMILE <h1> <label for="myfile">Select an image:</label> <input type="file" id="myfile" name="myfile">'
if __name__ == "__main__":
    app.run()
