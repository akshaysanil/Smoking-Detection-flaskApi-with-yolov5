# importing requirement models
import sys
import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response, Response
from werkzeug.exceptions import BadRequest
import os

# create flask app
app = Flask(__name__)

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

def detect_live(model):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert frame to PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        # run inference
        results = model(pil_img,size= 640)
        results.render()

        #encode image and yield it
        for img in results.ims:
            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ret, buffer = cv2.imencode('.jpg', RGB_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/',methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('detect_choice') == 'live':
            return Response(detect_live(dictOfModels[request.form.get('model_choice')]), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    else:
        # in the select we will have each key of the list in option
        return render_template('indexOG.html',len= len(listOfKeys), listOfKeys= listOfKeys)

def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest('missing file parameter!!!!')
    file = request.files['file']
    if file.filename == '':
        raise BadRequest('Given file is invalid...')
    return file

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
                model = dictOfModels[model_name]
                model.conf = 0.5


        for key in dictOfModels:
            listOfKeys.append(key)
    #starting app
    app.run(debug=True, host='0.0.0.0')
               
