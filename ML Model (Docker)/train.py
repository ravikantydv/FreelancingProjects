#!/usr/bin/python3
# train.py
# Xavier Vasques 13/04/2021

import platform
import sys
import numpy as np
import scipy
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from joblib import dump
from sklearn import preprocessing

def train():
    # Load directory paths for persisting model
    MODEL_PATH_LDA = "Lda.sav"
    MODEL_PATH_NN = "NN.sav"
    MODEL_PATH_RF = "RandomForest.sav"  # Path for the Random Forest model

    # Load, read and normalize training data
    training = "train.csv"
    data_train = pd.read_csv(training)

    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis=1)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)

    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')

    # Models training

    # Linear Discrimant Analysis (Default parameters)
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)

    # Serialize model
    dump(clf_lda, MODEL_PATH_LDA)

    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    clf_NN.fit(X_train, y_train)

    # Serialize model
    dump(clf_NN, MODEL_PATH_NN)

    # Random Forest Classifier
    clf_RF = RandomForestClassifier(n_estimators=100, random_state=0)  # Instantiate the RandomForestClassifier
    clf_RF.fit(X_train, y_train)

    # Serialize model
    dump(clf_RF, MODEL_PATH_RF)

if __name__ == '__main__':
    train()
