# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, request

from PIL import Image
import cv2

import numpy as np




from utils.rbg_to_xyz import rgb_to_xyz_convertor


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/', methods=['GET'])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello, World!abdallah'

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()


@app.route("/upload_image", methods=["POST"])
def upload_image():
    if 'image' in request.files:
        image_file = request.files['image']

        # Read the image using OpenCV
        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Converting from BGR to RGB
        rgb_to_xyz_convertor(img_cv2)
        # img = Image.open(image_file.stream)
        # img.show()
        return "Image received", 200
    else:
        return "No image received", 400