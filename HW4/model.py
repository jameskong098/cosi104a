"""
model.py

This module contains functions to train and evaluate a machine learning model.
It includes functions to train the model using cross-validation and to make predictions.

Functions:
- train_and_evaluate_model(X_train, y_train, feature_names): Trains and evaluates the model.
- make_predictions(model, X_test, output_file): Makes predictions on the test data and saves the results.
"""

import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import os
import json

def train_and_evaluate_model(X_train, y_train, feature_names):
    """
    Train a DecisionTreeClassifier with cross-validation and evaluate using F1 score.
    Optimize hyperparameters using GridSearchCV and save the best parameters to hyper_parameters.txt.
    Save the highest validation F1 score to highest_f1_score.txt and update it if the latest score is higher.

    Parameters:
    - X_train (ndarray): Preprocessed features of the training data.
    - y_train (Series): Labels of the training data.
    - feature_names (list): List of feature names.

    Returns:
    - model (DecisionTreeClassifier): Trained model.
    """
    print(f"\nTraining model with {X_train.shape[0]} samples and {X_train.shape[1]} features...\n")
    print(f"Features: {feature_names}")

    param_file = 'hyper_parameters.txt'
    if os.path.exists(param_file):
        with open(param_file, 'r') as file:
            best_params = json.load(file)
        print(f"\nLoaded best parameters from {param_file}: {best_params}")
    else:
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10]
        }
        model = DecisionTreeClassifier(random_state=42)
        f1_scorer = make_scorer(f1_score)
        print(f"\nPerforming GridSearchCV for best parameters...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=f1_scorer, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        with open(param_file, 'w') as file:
            json.dump(best_params, file)
        print(f"\nSaved best parameters to {param_file}: {best_params}")

    model = DecisionTreeClassifier(random_state=42, **best_params)
    f1_scorer = make_scorer(f1_score)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=f1_scorer)

    print(f'\nCross-Validation F1 Scores: {scores}')
    mean_f1_score = scores.mean()
    print(f'Mean F1 Score: {mean_f1_score}\n')

    # Save the highest validation F1 score
    highest_f1_file = 'highest_f1_score.txt'
    if os.path.exists(highest_f1_file):
        with open(highest_f1_file, 'r') as file:
            highest_f1_score = float(file.read())
    else:
        highest_f1_score = 0.0

    if mean_f1_score > highest_f1_score:
        with open(highest_f1_file, 'w') as file:
            file.write(str(mean_f1_score))
        highest_f1_score = mean_f1_score
        print(f"New highest validation F1 Score: {highest_f1_score}!!!\n")
    else:
        print(f"Highest validation F1 Score remains: {highest_f1_score}\n")

    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test, output_file):
    """
    Makes predictions on the test data and saves the results to a CSV file.

    Parameters:
    - model (DecisionTreeClassifier): Trained model.
    - X_test (ndarray): Preprocessed features of the test data.
    - output_file (str): Path to the output CSV file.

    Returns:
    - None
    """
    predictions = model.predict(X_test)
    output_df = pd.DataFrame({'isFraud': predictions})
    output_df.to_csv(output_file, index=False)
    