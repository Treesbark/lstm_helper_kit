#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `lstm_helper_kit` package."""

import pytest
from click.testing import CliRunner
from lstm_helper_kit.lstm_keras_model_class import LSTMKerasModel
from lstm_helper_kit import cli
import pandas as pd
import numpy as np


@pytest.fixture
def lstm_wrapper_object():
    """
    Creates an LSTM object for testing.

    Returns
    -------
    lstm_wrapper_object : lstm wrapper
        lstm wrapper object that is the main class in this package
    """
    lstm_wrapper_object = LSTMKerasModel()

    return lstm_wrapper_object

@pytest.fixture
def pandas_series_object():
    """
    Creates a test pandas series

    Returns
    -------
    pandas_series : pandas series
        pandas series for running tests over
    """
    data = np.array(np.arange(10))
    pandas_series = pd.Series(data)

    return pandas_series

def test_create_train_and_test_data(lstm_wrapper_object, pandas_series_object):
    train_set, test_set = lstm_wrapper_object.create_train_and_test_data(pandas_series_object)
    expected_training_array = np.array(np.arange(8)).reshape(8,1)
    expected_testing_array = np.array([8, 9]).reshape(2,1)

    assert (train_set==expected_training_array).all()
    assert (test_set==expected_testing_array).all()

def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'lstm_helper_kit.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output

