import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variable for model
model = None

def load_model():
    """Load the ML model with error handling"""
    global model
    try:
        model_path = "student_model.pkl"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False

        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for student marks"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Please check server logs."}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        hours = data.get('hours', None)
        if hours is None:
            return jsonify({"error": "Missing 'hours' in request"}), 400

        # Validate input
        try:
            hours = float(hours)
            if hours < 0:
                return jsonify({"error": "Hours cannot be negative"}), 400
        except ValueError:
            return jsonify({"error": "Hours must be a valid number"}), 400

        # Make prediction
        prediction = float(model.predict([[hours]])[0])
        logger.info(f"Prediction made for {hours} hours: {prediction}")

        return jsonify({
            "predicted_marks": prediction,
            "input_hours": hours,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Student Marks Prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "home": "/"
        }
    })

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        logger.error("Failed to load model. Exiting...")
        exit(1)

    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
