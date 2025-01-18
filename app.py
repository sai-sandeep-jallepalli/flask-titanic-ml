from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load model and label encoders
model_path = "pickle-models/train.pkl"
encoders_path = "pickle-models/label_encoders.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(encoders_path, 'rb') as file:
    label_encoders = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = []
    for key, encoder in label_encoders.items():
        if key in request.form:
            value = request.form[key]
            features.append(encoder.transform([value])[0])
        else:
            features.append(float(request.form[key]))
    prediction = model.predict([features])
    return render_template('index.html', prediction=f'Prediction: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)