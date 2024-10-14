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

def main():
    X_train, y_train, X_test = load_data('HW4_training.csv', 'HW4_test_input.csv')

    X_train, X_test = preprocess_data(X_train, X_test)

    # Train the model with cross-validation and evaluate
    model = train_and_evaluate_model(X_train, y_train)

    make_predictions(model, X_test, 'HW4_test_output.csv')

if __name__ == "__main__":
    main()