import pickle
from flask import Flask, render_template, request, app, jsonify, url_for
import numpy as np
import pandas as pd 


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.json['data']
    print(data)
    formatted_data = [list(data.values())]
    output = model.predict(formatted_data)[0]
    return jsonify(output)


@app.route('/predict_batch', methods=['POST'])
def predict_batch():

    batch = request.json['batch']
    print(batch)
    formatted_dat = list(batch.values())
    output1 = model.predict(formatted_dat)
    return jsonify(list(output1))


@app.route('/html_predict', methods=['POST'])
def html_predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    output = model.predict(final_features)[0]
    print(output)
    return render_template('home.html', prediction_text="fire prediction : {}".format(output))


if __name__=='__main__':
    app.run(debug=True)

