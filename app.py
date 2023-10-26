# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, request

from utils.RGB_convertor import conidtional_converting


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/', methods=['GET'])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello, World!'
print('app')

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if 'image' in request.files:

        image_file = request.files['image']
        conversion_mode = request.form['conversion_mode']

        conidtional_converting(image_file,conversion_mode)
    
        return "Image received", 200
    else:
        return "No image received", 400

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
