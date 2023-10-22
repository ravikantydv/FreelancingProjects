# We now need the json library so we can load and export json data
import json
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing

from flask import Flask

# Set environnment variables
MODEL_PATH_LDA = "Lda.sav"
MODEL_PATH_NN = "NN.sav"
MODEL_PATH_RF = "RandomForest.sav"  # Path for the Random Forest model

# Loading models
print("Loading model from: {}".format(MODEL_PATH_LDA))
inference_lda = load(MODEL_PATH_LDA)

print("Loading model from: {}".format(MODEL_PATH_NN))
inference_NN = load(MODEL_PATH_NN)

print("Loading model from: {}".format(MODEL_PATH_RF))
inference_RF = load(MODEL_PATH_RF)  # Load the Random Forest model

# Creation of the Flask app
app = Flask(__name__)

# API 1
@app.route('/line/<Line>')
def line(Line):
    with open('./test.json', 'r') as jsonfile:
        file_data = json.loads(jsonfile.read())
    return json.dumps(file_data[Line])

# API 2
@app.route('/prediction/<int:Line>', methods=['POST', 'GET'])
def prediction(Line):
    data = pd.read_json('./test.json')
    data_test = data.transpose()
    X = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis=1)
    X_test = X.iloc[Line, :].values.reshape(1, -1)
    
    prediction_lda = inference_lda.predict(X_test)
    prediction_nn = inference_NN.predict(X_test)
    
    return {'prediction LDA': int(prediction_lda), 'prediction Neural Network': int(prediction_nn)}

# API 3
@app.route('/score', methods=['POST', 'GET'])
def score():
    data = pd.read_json('./test.json')
    data_test = data.transpose()
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis=1)
    
    score_lda = inference_lda.score(X_test, y_test)
    score_nn = inference_NN.score(X_test, y_test)
    
    return {'Score LDA': score_lda, 'Score Neural Network': score_nn}

# API 4
@app.route('/prediction_rf/<int:Line>', methods=['POST', 'GET'])
def prediction_rf(Line):
    data = pd.read_json('./test.json')
    data_test = data.transpose()
    X = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis=1)
    X_test = X.iloc[Line, :].values.reshape(1, -1)
    
    prediction_rf = inference_RF.predict(X_test)  # Use the loaded Random Forest model for prediction
    
    return {'prediction Random Forest': int(prediction_rf)}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
