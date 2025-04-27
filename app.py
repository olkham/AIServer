# inference_server/
# ├── app.py                  # Main Flask application
# ├── config.py               # Configuration
# ├── plugins/                # Plugin directory
# │   ├── __init__.py         # Plugin registry
# │   ├── base.py             # Base plugin class
# │   ├── openvino_plugin.py  # Your existing OpenVINO plugin
# │   └── yolov8_plugin.py    # Example YOLOv8 plugin
# └── utils/                  # Utility functions


# app.py - Main Flask application
import datetime
import importlib
import inspect
from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from plugins import registry, PluginRegistry
import logging
from werkzeug.utils import secure_filename
import uuid

from plugins.base import InferencePlugin
from flask import render_template_string
import json
from flask import render_template_string, request, redirect, url_for, flash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='', static_folder='html_templates')

# Configuration
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load plugins
registry = PluginRegistry()
registry.discover_plugins()
logger.info(f"Discovered plugins: {registry.list_plugins()}")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# HTML template for the plugins dashboard
# Read dashboard template from file
with open('html_templates/dashboard.html', 'r') as file:
    DASHBOARD_TEMPLATE = file.read()

# Read HTML templates
with open('html_templates/upload_model.html', 'r') as file:
    UPLOAD_MODEL_TEMPLATE = file.read()

with open('html_templates/model_list.html', 'r') as file:
    MODEL_LIST_TEMPLATE = file.read()

with open('html_templates/model_detail.html', 'r') as file:
    MODEL_DETAIL_TEMPLATE = file.read()

with open('html_templates/inference.html', 'r') as file:
    INFERENCE_TEMPLATE = file.read()

# Model storage configuration
MODEL_FOLDER = './models'
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_INFO_FILE = os.path.join(MODEL_FOLDER, 'model_registry.json')

def get_model_registry():
    """Load the model registry from file or create if not exists"""
    if os.path.exists(MODEL_INFO_FILE):
        with open(MODEL_INFO_FILE, 'r') as f:
            return json.load(f)
    return {"models": []}

def save_model_registry(registry):
    """Save the model registry to file"""
    with open(MODEL_INFO_FILE, 'w') as f:
        json.dump(registry, f, indent=2)

@app.route('/gui/upload_model', methods=['GET', 'POST'])
def upload_model():
    global registry
    """Web interface for uploading models"""
    if request.method == 'POST':
        # Check if all required fields are present
        if 'model_file' not in request.files or not request.form.get('plugin_name'):
            return "Missing required fields", 400
        
        model_file = request.files['model_file']
        plugin_name = request.form['plugin_name']
        model_name = request.form.get('model_name', '')
        
        if model_file.filename == '':
            return "No file selected", 400
            
        if model_file:
            # Create directory for this model type if needed
            model_dir = os.path.join(MODEL_FOLDER, plugin_name)
            os.makedirs(model_dir, exist_ok=True)
            
            filename = secure_filename(model_file.filename)
            filepath = os.path.join(model_dir, filename)
            model_file.save(filepath)
            
            # Add to registry
            registry = get_model_registry()
            model_id = str(uuid.uuid4())
            model_info = {
                "id": model_id,
                "name": model_name or os.path.splitext(filename)[0],
                "plugin_name": plugin_name,
                "path": filepath,
                "uploaded_at": str(datetime.datetime.now())
            }
            registry["models"].append(model_info)
            save_model_registry(registry)
            
            return redirect(url_for('list_models'))
    
    # GET request shows the upload form
    plugins = registry.list_plugins()
    return render_template_string(UPLOAD_MODEL_TEMPLATE, plugins=plugins)

@app.route('/gui/models', methods=['GET'])
def list_models():
    """Web interface to list all uploaded models"""
    registry = get_model_registry()
    return render_template_string(MODEL_LIST_TEMPLATE, models=registry["models"])

@app.route('/gui/models/<model_id>', methods=['GET'])
def model_detail(model_id):
    """Web interface to show model details and load the model"""
    registry = get_model_registry()
    
    # Find the model by ID
    model = next((m for m in registry["models"] if m["id"] == model_id), None)
    if not model:
        return "Model not found", 404
    
    # Check if model is already loaded
    is_loaded = False
    try:
        plugin = registry.get_instance(model["plugin_name"], model_id)
        is_loaded = True
        model_metadata = plugin.metadata
    except ValueError:
        model_metadata = {}
    
    return render_template_string(MODEL_DETAIL_TEMPLATE, 
                                 model=model, 
                                 is_loaded=is_loaded,
                                 metadata=model_metadata)

@app.route('/gui/models/<model_id>/load', methods=['POST'])
def load_model_gui(model_id):
    """Web interface to load a model"""
    registry = get_model_registry()
    
    # Find the model by ID
    model = next((m for m in registry["models"] if m["id"] == model_id), None)
    if not model:
        return "Model not found", 404
    
    try:
        # Load the model with the plugin
        plugin = registry.load_plugin(
            model["plugin_name"], 
            model_id, 
            model["path"]
        )
        
        # Update model info with metadata
        for m in registry["models"]:
            if m["id"] == model_id:
                m["loaded"] = True
                m["metadata"] = plugin.metadata
        save_model_registry(registry)
        
        return redirect(url_for('model_detail', model_id=model_id))
    except Exception as e:
        return f"Error loading model: {str(e)}", 500

@app.route('/gui/inference/<model_id>', methods=['GET', 'POST'])
def run_inference_gui(model_id):
    """Web interface for running inference"""
    registry = get_model_registry()
    
    # Find the model by ID
    model = next((m for m in registry["models"] if m["id"] == model_id), None)
    if not model:
        return "Model not found", 404
    
    if request.method == 'POST':
        # Check if model is loaded first
        try:
            plugin = registry.get_instance(model["plugin_name"], model_id)
        except ValueError:
            return "Model not loaded. Please load the model first.", 400
            
        # Check if the post request has the file part
        if 'image' not in request.files:
            return "No image file provided", 400
            
        file = request.files['image']
        if file.filename == '':
            return "No image selected", 400
            
        if file and allowed_file(file.filename):
            # Save and process image
            img_filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            file.save(filepath)
            
            # Read image
            image = cv2.imread(filepath)
            if image is None:
                return "Could not read image", 400
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = plugin.predict(image)
            
            # Render results
            return render_template_string(INFERENCE_TEMPLATE,
                                         model=model,
                                         image_path=filepath,
                                         results=results)
    
    # GET request shows the upload form for inference
    return render_template_string(INFERENCE_TEMPLATE,
                                 model=model,
                                 image_path=None,
                                 results=None)


@app.route('/', methods=['GET'])
def dashboard():
    """Render the dashboard page"""
    global registry
    plugins = registry.list_plugins()
    return render_template_string(DASHBOARD_TEMPLATE, plugins=plugins)



# API routes
@app.route('/api/plugins', methods=['GET'])
def list_plugins():
    """List all available plugins"""
    plugins = registry.list_plugins()
    return jsonify({"plugins": plugins})

@app.route('/api/models/load', methods=['POST'])
def load_model():
    """Load a model using a specific plugin"""
    data = request.json
    
    if not data or 'plugin_name' not in data or 'model_path' not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    plugin_name = data['plugin_name']
    model_path = data['model_path']
    model_id = data.get('model_id', str(uuid.uuid4()))
    
    try:
        # Additional parameters for model loading
        kwargs = {k: v for k, v in data.items() if k not in ['plugin_name', 'model_path', 'model_id']}
        
        # Load the model
        plugin = registry.load_plugin(plugin_name, model_id, model_path, **kwargs)
        
        return jsonify({
            "status": "success",
            "message": f"Model loaded successfully",
            "model_id": model_id,
            "plugin": plugin.name,
            "metadata": plugin.metadata
        })
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/inference/<plugin_name>/<model_id>', methods=['POST'])
def run_inference(plugin_name, model_id):
    """Run inference on an image using a loaded model"""
    try:
        # Check if model is loaded
        plugin = registry.get_instance(plugin_name, model_id)
        
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        if file and allowed_file(file.filename):
            # Read and process image
            img_filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            file.save(filepath)
            
            # Read image
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({"error": "Could not read image"}), 400
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            original_size = image.shape[:2]  # (height, width)
            results = plugin.predict(image)
            
            # Add metadata to results
            response = {
                "results": results,
                "metadata": {
                    "plugin": plugin_name,
                    "model_id": model_id,
                    "image_size": {
                        "height": original_size[0],
                        "width": original_size[1]
                    }
                }
            }
            
            return jsonify(response)
        
        return jsonify({"error": "Invalid file format"}), 400
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Example custom plugin registration route
@app.route('/api/plugins/register', methods=['POST'])
def register_custom_plugin():
    """
    Register a custom plugin module
    This is a simple implementation - in production, you'd want more security
    """
    data = request.json
    
    if not data or 'module_path' not in data:
        return jsonify({"error": "Missing module_path parameter"}), 400
    
    try:
        module_path = data['module_path']
        module = importlib.import_module(module_path)
        
        # Find plugin classes in the module
        found_plugins = []
        for item_name, item in inspect.getmembers(module, inspect.isclass):
            if issubclass(item, InferencePlugin) and item != InferencePlugin:
                registry.register_plugin(item)
                found_plugins.append(item_name)
        
        if not found_plugins:
            return jsonify({"error": "No valid plugins found in module"}), 400
        
        return jsonify({
            "status": "success", 
            "registered_plugins": found_plugins
        })
        
    except ImportError as e:
        return jsonify({"error": f"Could not import module: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the application
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=555)