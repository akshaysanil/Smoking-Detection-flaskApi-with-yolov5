# importing requirement models
import sys
import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response
from werkzeug.exceptions import BadRequest
import os

# create flask app
app = Flask(__name__)
 

# Here a python dictionary is created to store the name of each of your weight files (key) and the corresponding model (value).

# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {}


# create a list of keys to use them in the select part of the html code
listOfKeys = []


# write the interface function
def get_prediction(img_bytes,model):
    img = Image.open(io.BytesIO(img_bytes))

    # inference
    results = model(img,size= 640)
    return results


# Define the GET method
@app.route('/',methods=['GET'])
def get():
    # in the select we will have each key of the list in option
    return render_template('index.html',len= len(listOfKeys), listOfKeys= listOfKeys)


# Define the POST method
@app.route('/',methods=['POST'])
def predict():
    file = extract_img(request)
    img_bytes = file.read()

    #choice of the model
    results = get_prediction(img_bytes,dictOfModels[request.form.get('model_choice')])
    print(f"User selected model : {request.form.get('model_choice')}")

    #update results.image with boxes and labels
    results.render()

    #encoding the resulting image and render it
    for img in results.ims:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_arr = cv2.imencode('.jpg',RGB_img)[1]
        response = make_response(im_arr.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
    return response


def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest('missing file parameter!!!!')
    file = request.files['file']
    if file.filename == '':
        raise BadRequest('Given file is invalid...')
    return file


#Define the main and start the app:
if __name__ == '__main__':
    print('starting yolov5 webservice....')
    #getting dir containing models from comand args (or default 'model_train')
    models_dir = 'models_train'
    if len(sys.argv) > 1:
        models_dir = sys.argv[1]
    print(f'watcing for yolov5 models under {models_dir}...')
    for r, d, f in os.walk(models_dir):
        for file in f:
            if '.pt' in file:
                #example : file = 'model1.pt'
                #the path of each model: os.path.join(r,file)
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(r,file)
                print(f'Loading model{model_path} with path {model_path}...')
                dictOfModels[model_name] = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, force_reload=True)
                threshold = 0.6
        for key in dictOfModels:
            listOfKeys.append(key)

    #starting app
    app.run(debug=True, host='0.0.0.0')

    


