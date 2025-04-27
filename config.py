# config.py - Configuration settings for the inference server

import os
from pathlib import Path

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

# Server settings
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))
API_PREFIX = "/api"

# Upload settings
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16MB
TEMP_FOLDER = os.environ.get("TEMP_FOLDER", os.path.join(BASE_DIR, "temp"))

# Plugin settings
PLUGINS_DIR = os.environ.get("PLUGINS_DIR", os.path.join(BASE_DIR, "plugins"))
PLUGIN_DISCOVERY_ENABLED = os.environ.get("PLUGIN_DISCOVERY_ENABLED", "True").lower() == "true"
CUSTOM_PLUGIN_REGISTRATION_ENABLED = os.environ.get("CUSTOM_PLUGIN_REGISTRATION_ENABLED", "False").lower() == "true"

# Model settings
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("DEFAULT_CONFIDENCE_THRESHOLD", 0.5))
MODEL_CACHE_SIZE = int(os.environ.get("MODEL_CACHE_SIZE", 5))  # Number of models to keep in memory

# Device settings
DEFAULT_DEVICE = os.environ.get("DEFAULT_DEVICE", "CPU")  # CPU, GPU, etc.
AUTO_DEVICE_SELECTION = os.environ.get("AUTO_DEVICE_SELECTION", "True").lower() == "true"

# Logging settings
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = os.environ.get("LOG_FILE", os.path.join(BASE_DIR, "logs", "inference_server.log"))

# Security settings
ENABLE_AUTH = os.environ.get("ENABLE_AUTH", "False").lower() == "true"
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")

# Result caching settings
ENABLE_RESULT_CACHING = os.environ.get("ENABLE_RESULT_CACHING", "False").lower() == "true"
RESULT_CACHE_TTL = int(os.environ.get("RESULT_CACHE_TTL", 3600))  # Time to live in seconds

# Performance settings
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 1))

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Plugin-specific configurations
PLUGIN_CONFIGS = {
    "OpenVINOPlugin": {
        "enable_preprocessing_cache": os.environ.get("OPENVINO_ENABLE_PREPROC_CACHE", "False").lower() == "true",
        "num_streams": int(os.environ.get("OPENVINO_NUM_STREAMS", 1)),
        "num_threads": int(os.environ.get("OPENVINO_NUM_THREADS", 0)),  # 0 means auto
    },
    "YOLOv8Plugin": {
        "conf_threshold": float(os.environ.get("YOLO_CONF_THRESHOLD", 0.25)),
        "iou_threshold": float(os.environ.get("YOLO_IOU_THRESHOLD", 0.45)),
    },
    # Add configurations for other plugins here
}

# API rate limiting
ENABLE_RATE_LIMITING = os.environ.get("ENABLE_RATE_LIMITING", "False").lower() == "true"
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", 100))  # requests per hour

# Feature flags
FEATURE_PREPROCESSING_CUSTOMIZATION = os.environ.get("FEATURE_PREPROCESSING_CUSTOMIZATION", "True").lower() == "true"
FEATURE_MODEL_VERSIONING = os.environ.get("FEATURE_MODEL_VERSIONING", "False").lower() == "true"
FEATURE_BATCH_INFERENCE = os.environ.get("FEATURE_BATCH_INFERENCE", "False").lower() == "true"

# Miscellaneous
DEFAULT_RESPONSE_FORMAT = os.environ.get("DEFAULT_RESPONSE_FORMAT", "json")  # json, xml, etc.
ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "False").lower() == "true"