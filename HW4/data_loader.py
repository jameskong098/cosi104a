"""
data_loader.py

This module contains functions to load training and test data from CSV files.
"""

import pandas as pd

def load_data(train_path, test_path):
    """
    Load training and test data from CSV files.

    Parameters:
    - train_path (str): Path to the training data CSV file.
    - test_path (str): Path to the test data CSV file.

    Returns:
    - X_train (DataFrame): Features of the training data.
    - y_train (Series): Labels of the training data.
    - X_test (DataFrame): Features of the test data.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop('isFraud', axis=1)
    y_train = train_data['isFraud']
    X_test = test_data

    return X_train, y_train, X_test
