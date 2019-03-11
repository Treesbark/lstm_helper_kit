# -*- coding: utf-8 -*-

"""Main module."""

# Importing the libraries

import matplotlib.pyplot as plt
import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model, model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
plt.style.use('fivethirtyeight')


def save_keras_model(file_save_path, model, force_overwrite=False):
    """
    Method for saving a Keras moodel that checks to ensure the file does not already exist

    Parameters
    ----------
    file_save_path : str
        the entire path for where the model should be saved
    force_overwrite : bool
        boolean to see if the whole string should be overwritten

    Returns
    -------
    string
        a value in a string

    """
    from pathlib import Path

    # Checks to make sure a '.json' is appended to the end of the string
    if file_save_path[-4:] != ".json":
        file_save_path = file_save_path + ".json"

    # Check to see if file is in path to avoid overwriting
    file_checker = Path(file_save_path)
    if file_checker.is_file() or force_overwrite == False:
        print("File already exists")
    else:
        # Save the model to json
        model_json = model.to_json()
        with open(file_save_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model " + str(file_save_path))


def load_keras_model(complete_model_path):
    """
    Method for loading a Keras model

    Parameters
    ----------
    complete_model_path : str
        the entire path for where the model is saved

    Returns
    -------
    Keras model
        a loaded Keras model

    """
    # Checks to make sure a '.json' is appended to the end of the string
    if complete_model_path[-4:] != ".json":
        complete_model_path = complete_model_path + ".json"

    # Check to see if the model exists
    file_checker = Path(complete_model_path)
    if file_checker.is_file() == False:
        print("Model not found")
    else:
        # load json and create model
        json_file = open(complete_model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded Keras model from " + complete_model_path)
        return loaded_model



class LSTMKerasModel():
    """A class for abstracting out several functions in a Keras Model"""

    def __init__(self):
        self.model = Sequential()
        self.data_scaler = MinMaxScaler(feature_range=(0,1))

    def load_keras_model(self, complete_model_path):
        """
        Method for loading a Keras model

        Parameters
        ----------
        complete_model_path : str
            the entire path for where the model is saved

        Returns
        -------
        Keras model
            a loaded Keras model

        """
        from pathlib import Path

        # Checks to make sure a '.json' is appended to the end of the string
        if complete_model_path[-4:] != ".json":
            complete_model_path = complete_model_path + ".json"

        # Check to see if the model exists
        file_checker = Path(complete_model_path)
        if file_checker.is_file() == False:
            print("Model not found")
        else:
            # load json and create model
            json_file = open(complete_model_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model.h5")
            print("Loaded Keras model from " + complete_model_path)
            self.model = loaded_model

    def save_keras_model_to_file(self, file_save_path, force_overwrite=False):
        """
        Method for saving a Keras model that checks to ensure the file does not already exist

        Parameters
        ----------
        file_save_path : str
            the entire path for where the model should be saved
        force_overwrite : bool
            boolean to see if the whole string should be overwritten

        Returns
        -------
        string
            a value in a string

        """
        from pathlib import Path

        # Checks to make sure a '.json' is appended to the end of the string
        if file_save_path[-4:] != ".json":
            file_save_path = file_save_path + ".json"

        # Check to see if file is in path to avoid overwriting
        file_checker = Path(file_save_path)
        if file_checker.is_file() or force_overwrite == False:
            print("File already exists")
        else:
            # Save the model to json
            model_json = self.model.to_json()
            with open(file_save_path, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights("model.h5")
            print("Saved model " + str(file_save_path))

    def save_keras_model_to_class(self, model):
        """
        Saves the created Keras model to the instantiated model object for easy of use and encapsulation

        Parameters
        ----------
        model : Keras Model
            the model that was trained and is to be wrapped in this abstract class

        """
        self.model = model

    def predict_point_by_point(self, input_data, window_size, test_data_length):
        """
        Predicts the future values point by point with known data

        Parameters
        ----------
        input_data : numpy array
            the data on which to build prediction - must be as long or longer than window size
        window_size : int
            the size of the window the model expects to ingest into the LSTM
        test_data_length : int
            the length of test data on which to predict

        Returns
        -------
        numpy array
            the future values that were predicted point-by-point

        """
        X_test = []

        # Predict the future values step-by-step
        for i in range(window_size, test_data_length):
            X_test.append(input_data[i - window_size:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        final_output_array = self.model.predict(X_test)
        final_output_array = self.data_scaler.inverse_transform(final_output_array)
        return final_output_array

    def predict_future_sequence_for_given_time_length(self, input_data, window_size, time_to_predict_into_future):
        """
        Predicts future sequences of values based upon an initial array array passed to the model

        Parameters
        ----------
        input_data : numpy array
            the data on which to build prediction - must be as long or longer than window size
        window_size : int
            the size of the window the model expects to ingest into the LSTM
        time_to_predict_into_future : int
            the length of time the LSTM will predict into the future the outputs

        Returns
        -------
        numpy array
            the future predicted values

        """
        # Instantiate the array that will be used to predict values
        predictor_array = input_data[0:window_size, 0]

        # Instantiate the final output array
        final_output_array = []

        # Predict future value for time perido in future provided
        for i in range(time_to_predict_into_future):
            # Save the array that is about to be used to predict the outcome but slice the first value
            next_step_array = predictor_array[1:]

            # Prepare the array to be fed to the LSTM
            predictor_array = np.array([predictor_array])
            predictor_array = np.reshape(predictor_array,
                                         (predictor_array.shape[0], predictor_array.shape[1], 1))
            predicted_output = self.model.predict(predictor_array)

            # Append the predicted output to the next array to be used in the NN
            predictor_array = np.append(next_step_array, predicted_output)

            # Throw the predicted output onto the end of the output array
            final_output_array = np.append(final_output_array, predicted_output)

        final_output_array = self.data_scaler.inverse_transform([final_output_array])
        final_output_array = final_output_array[0][:, newaxis]
        return final_output_array