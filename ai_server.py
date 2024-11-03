# Start the server:
# 	python3 ai_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@images/test-image.jpg 'http://<server>:<port>/v1/vision/detection'

# Customer AI server for AgentDVR or similar that uses the same API as ProjectCode
# This server is a simple Flask server that listens for POST requests with an image file attached
# The server will then run the image through the Geti model and return the results as a JSON object
# https://intel.ly/4hjJGpw

# ========================================================================================================
#   Don't consider this server as a production-ready server, it's just for my personal use and testing
# ========================================================================================================

import datetime
import uuid
import cv2
from geti_sdk import Geti
from geti_sdk.deployment import Deployment
import os
import flask
import numpy as np
import time
from dotenv import load_dotenv

app = flask.Flask(__name__)


DEFAULT_MODELS_DIRECTORY = 'models'

ROOT_URL = '/v1/vision/detection'
LIST_URL = '/v1/vision/custom/list'

@app.route('/')
def info():
    info_str = 'Flask app exposing Geti model'
    return info_str

def prepare_request_data(type='detection'):
    if type == 'detection':
        data = {
            "message": "",
            "count": 0,
            "predictions": [],
            "success": False,
            "processMs": -1,
            "inferenceMs": -1,
            "moduleId": "Intel Geti SDK",
            "moduleName": "Intel Geti",
            "code": 200,
            "command": "detect",
            "requestId": str(uuid.uuid4()),
            "inferenceDevice": device,
            "analysisRoundTripMs": -1,
            "processedBy": flask.request.host,
            "timestampUTC": datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        }
    elif type == 'custom':
        data = {
            "success": False,
            "models": [],
            "moduleId": "Intel Geti SDK",
            "moduleName": "Intel Geti",
            "code": 200,
            "command": "list-custom",
            "requestId": str(uuid.uuid4()),
            "inferenceDevice": device,
            "analysisRoundTripMs": -1,
            "processedBy": flask.request.host,
            "timestampUTC": datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        }

    return data

@app.route(LIST_URL, methods=['POST', 'GET'])
def list_models():
    
    list_of_models = os.listdir(DEFAULT_MODELS_DIRECTORY)
    models = []
    for model in list_of_models:
        model_path = os.path.join(DEFAULT_MODELS_DIRECTORY, model)
        if os.path.isdir(model_path):
            models.append(model)
    
    response = prepare_request_data(type='custom')
    response["models"] = models

    
    return flask.jsonify(response)


@app.route('/v1/vision/custom/<model_name>', methods=['POST'])
def custom_predict(model_name):
    model_name = flask.request.view_args.get('model_name')
    if model_name not in deployments:
        project_name = model_name
        deployment = geti.deploy_project(project_name=project_name, output_folder=project_name)
        deployment.load_inference_models(device=device)
        deployments[model_name] = deployment
        
    req_start = time.time()
    data = prepare_request_data(type='detection')

    if flask.request.method == 'POST':
        if flask.request.form.get('min_confidence'):
            threshold=float(flask.request.form['min_confidence'])
        else:
            threshold=float(0.4)
        if flask.request.files.get('image'):
            image_file = flask.request.files['image']
            image_bytes = image_file.read()
            # image = Image.open(io.BytesIO(image_bytes))

            frame_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inf_start = time.time()
            prediction = deployments[model_name].infer(image=frame_rgb)
            inf_duration = time.time() - inf_start

            if len(prediction.annotations) > 0:
                data['success'] = True
                preds = []

                for obj in prediction.annotations:
                    
                    if float(obj.labels[0].probability) >= float(threshold):
                        preds.append(
                            {
                                "confidence": float(obj.labels[0].probability),
                                "label": obj.labels[0].name,
                                "x_min": int(obj.shape.x),
                                "y_min": int(obj.shape.y),
                                "x_max": int(obj.shape.x + obj.shape.width),
                                "y_max": int(obj.shape.y + obj.shape.height)
                            }
                        )
                data['predictions'] = preds
                data['count'] = len(preds)
                data['message'] = 'Objects detected'
                data['processMs'] = round((time.time() - req_start) * 1000)
                data['inferenceMs'] = round(inf_duration * 1000)
                data['analysisRoundTripMs'] = data['processMs']
                
                print('Objects detected:', len(preds))
                print('Objects:', preds)
            else:
                print('No objects detected')

    # return the data dictionary as a JSON response
    return flask.jsonify(data)        


@app.route(ROOT_URL, methods=['POST'])
def predict():
    req_start = time.time()
    data = prepare_request_data(type='detection')

    if flask.request.method == 'POST':
        if flask.request.form.get('min_confidence'):
            threshold=float(flask.request.form['min_confidence'])
        else:
            threshold=float(0.4)
        if flask.request.files.get('image'):
            image_file = flask.request.files['image']
            image_bytes = image_file.read()
            # image = Image.open(io.BytesIO(image_bytes))

            frame_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inf_start = time.time()
            prediction = det_deployment.infer(image=frame_rgb)
            inf_duration = time.time() - inf_start

            if len(prediction.annotations) > 0:
                data['success'] = True
                preds = []

                for obj in prediction.annotations:
                    
                    if float(obj.labels[0].probability) >= float(threshold):
                        preds.append(
                            {
                                "confidence": float(obj.labels[0].probability),
                                "label": obj.labels[0].name,
                                "x_min": int(obj.shape.x),
                                "y_min": int(obj.shape.y),
                                "x_max": int(obj.shape.x + obj.shape.width),
                                "y_max": int(obj.shape.y + obj.shape.height)
                            }
                        )
                data['predictions'] = preds
                data['count'] = len(preds)
                data['message'] = 'Objects detected'
                data['processMs'] = round((time.time() - req_start) * 1000)
                data['inferenceMs'] = round(inf_duration * 1000)
                data['analysisRoundTripMs'] = data['processMs']
                
                print('Objects detected:', len(preds))
                print('Objects:', preds)
            else:
                print('No objects detected')

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == '__main__':
    # Load environment variables from .env file
    load_dotenv()

    # Read PAT and server from environment variables
    pat = os.getenv('GETI_PAT')
    server = os.getenv('GETI_SERVER')
    verify_cert = False
    proxies = None

    # connect to Get
    geti = None
    try:
        geti = Geti(host=server, token=pat, verify_certificate=verify_cert, proxies=proxies)
    except Exception as e:
        print(f"Error connecting to Geti: {str(e)}")

    #urgh this is ugly but it works # TODO: improve this
    global det_deployment, device, deployments
    
    deployments: dict[str, Deployment] = {}
    device = 'CPU'
    det_project_name = 'kitchen'
    
    # Load the detection model if it exists
    if os.path.exists(os.path.join(DEFAULT_MODELS_DIRECTORY, det_project_name)):
        det_deployment = Deployment.from_folder(os.path.join(DEFAULT_MODELS_DIRECTORY, det_project_name))
    else:
        #otherwise pull the model from Geti
        if not geti:
            raise Exception("Geti not connected and model not found locally")
        det_deployment = geti.deploy_project(project_name=det_project_name, output_folder=det_project_name)
    
    # Load the model into memory
    det_deployment.load_inference_models(device=device)
    deployments[det_project_name] = det_deployment

    # start the flask app
    app.run(host='0.0.0.0', debug=False, port=8080)