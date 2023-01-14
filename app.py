from flask import Flask,render_template,request
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array, load_img
import cv2
import os
import os.path as op
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/images/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# dimensions of images
img_width, img_height = (64, 64)
classes = ['man','woman']
# load model
model = load_model('model_0.968.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
loop_running = False

def getPrediction(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return 'female' if model.predict(img) == 0 else 'male'

@app.route("/")
def index():
  return render_template("index.html")


@app.route('/gender_from_image', methods=['POST'])
def get_gender_from_image():
    if request.method == 'POST':
        file = request.files['test_image']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join('static/images', filename))
            path = op.join('static/images', filename)
    # remove the image from the static/images folder
    os.remove(path)

    return render_template('image_response.html', gender=getPrediction(path), image_path=path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)




