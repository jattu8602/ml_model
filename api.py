import os
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    hours = data.get('hours', None)
    if hours is None:
        return jsonify({"error": "Missing 'hours' in request."}), 400
    prediction = float(model.predict([[hours]])[0])
    return jsonify({"predicted_marks": prediction})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
