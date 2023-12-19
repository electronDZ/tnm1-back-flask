# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, request

from utils.RGB_convertor import conidtional_converting
from filters.noise import gaussian_noise
from filters.noise import pepper_salt_noise
from filters.prewit import  prewitt
from filters.robert import  apply_roberts_filter
from filters.sobel import  apply_sobel_filter_manual
from filters.seuilSim import  SeuilSim
from filters.seuilHil import  SeuilHys


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route("/", methods=["GET"])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return "Hello, World!"


@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "image" in request.files:
        image_file = request.files["image"]
        conversion_mode = request.form["conversion_mode"]
        # gaussian_noise(image_file)
        conidtional_converting(image_file,conversion_mode)

        return "Image received", 200
    else:
        return "No image received", 400


@app.route("/noise", methods=["POST"])
def upload_noise():
    if "image" in request.files:
        image_file = request.files["image"]
        conversion_mode = request.form["conversion_mode"]
        value = request.form["value"]
        if conversion_mode == 'gaussien':
            # prewitt(image_file)
            # apply_roberts_filter(image_file)
            # apply_sobel_filter_manual(image_file)
            # SeuilSim(image_file)
            SeuilHys(image_file)
        else:
            pepper_salt_noise(image_file,  int(value),  int(value))
            
            

        return "Image received", 200
    else:
        return "No image received", 400


# @app.route("/noise", methods=["POST"])
# def upload_noise():
#     if "image" in request.files:
#         image_file = request.files["image"]
#         conversion_mode = request.form["conversion_mode"]
#         value = request.form["value"]
#         if conversion_mode == 'gaussien':
#             gaussian_noise(image_file, int(value))
#         else:
#             pepper_salt_noise(image_file,  int(value),  int(value))
            
            

#         return "Image received", 200
#     else:
#         return "No image received", 400


# main driver function
if __name__ == "__main__":
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
