from flask import Flask,render_template,request
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
import os
import cv2
import cvlib as cv
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# dimensions of images
img_width, img_height = (64, 64)
classes = ['man','woman']
# load model
model = load_model('model_0.968.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
loop_running = False

def get_gender(frame):
    # load and resize image to 64x64
    frame = cv2.resize(frame, (img_width, img_height))
    # convert frame to numpy array
    frame = img_to_array(frame)
    # expand dimension of frame
    frame = np.expand_dims(frame, axis=0)
    print(model.predict(frame)[0][0])
    # making prediction with model
    return 'female' if model.predict(frame)[0][0] < 0.5 else 'male'


def output_image(image_path):
    predictions = []
    global n_faces
    image = cv2.imread(image_path)
    # apply face detection
    face, confidence = cv.detect_face(image)

    if len(face) <= 0:
        print('No face detected')
    # loop through detected faces
    else:
        print(f'{len(face)} faces detected')
        n_faces = len(face)
    for f in face:
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(image[startY:endY,startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        
        Y = startY - 10 if startY > 20 else startY + 10
        prediction = get_gender(face_crop)
        print(prediction)
        predictions.append(prediction)
        cv2.putText(image, prediction, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)



    if(cv2.imwrite('static/images/detected_faces.jpg', image) == False):
        print("Error saving the image")
    else:
        print('image saved')
    return 'static/images/detected_faces.jpg'


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
            path = os.path.join('static/images', filename)
    # remove the image from the static/images folder
    return render_template('image_response.html', image_path=output_image(path), number_of_people=n_faces)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)




