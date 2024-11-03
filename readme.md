# Custom AI Server for AgentDVR

This is a simple Flask server that listens for POST requests with an image file attached. The server runs the image through the Geti model and returns the results as a JSON object. This server is designed for personal use and testing.


⚠️ **Disclaimer:** This documentation is mostly written by GitHub Copilot and may require further review and adjustments. ⚠️


## Prerequisites

- Python 3.x
- `pip` (Python package installer)
- `virtualenv` (optional but recommended)

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add the following environment variables:
    ```env
    GETI_PAT=<your_geti_pat>
    GETI_SERVER=<your_geti_server>
    ```

## Starting the Server

1. Run the server:
    ```sh
    python ai_server.py
    ```

2. The server will start on `http://0.0.0.0:8080`.

## API Endpoints

### Root Endpoint

- **URL:** `/`
- **Method:** `GET`
- **Description:** Returns a simple info string.

### List Models

- **URL:** `/v1/vision/custom/list`
- **Method:** `POST` or `GET`
- **Description:** Lists all available models in the `models` directory.
- **Response:**
    ```json
    {
        "success": true,
        "models": ["model1", "model2"],
        "moduleId": "Intel Geti SDK",
        "moduleName": "Intel Geti",
        "code": 200,
        "command": "list-custom",
        "requestId": "unique-request-id",
        "inferenceDevice": "CPU",
        "analysisRoundTripMs": -1,
        "processedBy": "localhost",
        "timestampUTC": "Tue, 22 Oct 2024 19:42:58 GMT"
    }
    ```

### Custom Predict

- **URL:** `/v1/vision/custom/<model_name>`
- **Method:** `POST`
- **Description:** Runs inference on the provided image using the specified model.
- **Request:**
    - **Form Data:**
        - `image`: The image file to be processed.
        - `min_confidence` (optional): Minimum confidence threshold for predictions.
- **Response:**
    ```json
    {
        "message": "Objects detected",
        "count": 1,
        "predictions": [
            {
                "confidence": 0.6794257760047913,
                "label": "cat",
                "x_min": 0,
                "y_min": 459,
                "x_max": 196,
                "y_max": 696
            }
        ],
        "success": true,
        "processMs": 337,
        "inferenceMs": 337,
        "moduleId": "ObjectDetectionYOLOv5-6.2",
        "moduleName": "Object Detection (YOLOv5 6.2)",
        "code": 200,
        "command": "detect",
        "requestId": "unique-request-id",
        "inferenceDevice": "CPU",
        "analysisRoundTripMs": 420,
        "processedBy": "localhost",
        "timestampUTC": "Tue, 22 Oct 2024 18:30:14 GMT"
    }
    ```

### Predict

- **URL:** `/v1/vision/detection`
- **Method:** `POST`
- **Description:** Runs inference on the provided image using the default detection model.
- **Request:**
    - **Form Data:**
        - `image`: The image file to be processed.
        - `min_confidence` (optional): Minimum confidence threshold for predictions.
- **Response:** Same as the Custom Predict endpoint.

## Notes

- This server is not production-ready and is intended for personal use and testing.
- Ensure that the `models` directory contains the necessary models for inference.
- The server uses the Intel Geti SDK for model deployment and inference.