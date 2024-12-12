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

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from data_loader import load_data
from preprocessor import preprocess_data
from model import train_and_evaluate_model, make_predictions
from timer import start_timer, get_time_passed

def main():
    start_time = start_timer()

    X_train, y_train, X_test, test_ids = load_data('train.csv', 'test.csv', 'series_train.parquet', 'series_test.parquet')

    X_train, X_test, feature_names = preprocess_data(X_train, X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    best_model, models = train_and_evaluate_model(X_train, y_train, feature_names)

    val_predictions = best_model.predict(X_val)

    kappa = cohen_kappa_score(y_val, val_predictions, weights='quadratic')
  
    print(f"Validation Quadratic Weighted Kappa: {kappa}")

    make_predictions(models, X_test, 'submission.csv', test_ids)

    get_time_passed(start_time)

if __name__ == "__main__":
    main()