from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("MODEL.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])

def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)[0]
        return render_template('index.html', prediction = prediction)
    except Exception as e:
        return render_template('index.html', prediction = f"Error : {str(e)}")

if __name__ == "__main__":
    app.run(debug = True)