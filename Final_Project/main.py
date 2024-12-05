"""
Author: James Kong
Course: COSI 104A - Introduction to Machine Learning
Assignment: Final Project
Date: 12/13/2024

Description:
This script orchestrates the data loading, preprocessing, model training, and prediction steps.
It loads the training and test data, preprocesses them, trains a model using cross-validation,
and predicts fraudulent transactions within the test data, saving the results to a CSV file.
"""

from data_loader import load_data
from preprocessor import preprocess_data
from model import select_best_algorithm, train_and_evaluate_model, make_predictions
from timer import start_timer, get_time_passed

def main():
    start_time = start_timer()

    X_train, y_train, X_test = load_data('train.csv', 'test.csv')

    X_train, X_test, feature_names = preprocess_data(X_train, X_test)

    select_best_algorithm(X_train, y_train, True)
    
    model = train_and_evaluate_model(X_train, y_train, feature_names)

    make_predictions(model, X_test, 'sample_submission.csv')

    get_time_passed(start_time)

if __name__ == "__main__":
    main()
    