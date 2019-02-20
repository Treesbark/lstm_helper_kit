# -*- coding: utf-8 -*-

"""Main module."""

# Importing the libraries

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from keras.models import model_from_json


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
        return loaded_model