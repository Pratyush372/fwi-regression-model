from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))


@app.route('/', methods=['GET'])
def home():
    # First load → no prediction
    return render_template('index.html', result=None)


@app.route('/predictdata', methods=['POST'])
def predict_data_point():

    # Collect input values from the form
    temperature = float(request.form['Temperature'])
    rh = float(request.form['RH'])
    ws = float(request.form['Ws'])
    rain = float(request.form['Rain'])
    ffmc = float(request.form['FFMC'])
    dmc = float(request.form['DMC'])
    isi = float(request.form['ISI'])
    classes = int(request.form['Classes'])
    region = int(request.form['Region'])

    # Create a numpy array for the input
    input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])

    # Scale input using the scaler
    scaled_input = standard_scaler.transform(input_data)

    # Predict using the model
    prediction = ridge_model.predict(scaled_input)[0]

    # Render the same page with result
    return render_template('index.html', result=prediction)


if __name__ == "__main__":
    app.run(debug=True)
