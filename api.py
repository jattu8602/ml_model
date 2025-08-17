# api.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    hours = data['hours']
    prediction = model.predict([[hours]])[0]
    return jsonify({"predicted_marks": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
