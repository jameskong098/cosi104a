"""
Author: James Kong
Course: COSI 116A - Introduction to Machine Learning
Assignment: HW4
Date: 10/17/24

Description:
This script orchestrates the data loading, preprocessing, model training, and prediction steps.
It loads the training and test data, preprocesses them, trains a model using cross-validation,
and predicts fraudulent transactions within the test data, saving the results to a CSV file.
"""

from data_loader import load_data
from preprocessor import preprocess_data
from model import train_and_evaluate_model, make_predictions
from timer import start_timer, get_time_passed

def main():
    start_time = start_timer()

    X_train, y_train, X_test = load_data('HW4_training.csv', 'HW4_test_input.csv')

    X_train, X_test, feature_names = preprocess_data(X_train, X_test)

    # Train the model with cross-validation and evaluate
    model = train_and_evaluate_model(X_train, y_train, feature_names)

    make_predictions(model, X_test, 'HW4_test_output.csv')

    get_time_passed(start_time)

if __name__ == "__main__":
    main()