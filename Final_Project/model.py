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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import os
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def select_best_algorithm(X_train, y_train, run=False):
    algorithms = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    }

    best_score = 0
    best_algorithm = None

    for name, algorithm in algorithms.items():
        scores = cross_val_score(algorithm, X_train, y_train, cv=5, scoring='f1_weighted')
        mean_score = scores.mean()
        print(f'{name} F1 Score: {mean_score}')

        if mean_score > best_score:
            best_score = mean_score
            best_algorithm = algorithm

    print(f'Best Algorithm: {best_algorithm.__class__.__name__} with F1 Score: {best_score}')
    return best_algorithm

def train_and_evaluate_model(X_train, y_train, feature_names):
    """
    Train a RandomForestClassifier with cross-validation and evaluate using F1 score.
    Optimize hyperparameters using GridSearchCV and save the best parameters to hyper_parameters.txt.
    Save the highest validation F1 score to highest_f1_score.txt and update it if the latest score is higher.

    Parameters:
    - X_train (ndarray): Preprocessed features of the training data.
    - y_train (Series): Labels of the training data.
    - feature_names (list): List of feature names.

    Returns:
    - model (RandomForestClassifier): Trained model.
    """
    print(f"\nTraining model with {X_train.shape[0]} samples and {X_train.shape[1]} features...\n")
    print(f"Features: {feature_names}")

    param_file = 'hyper_parameters.txt'
    highest_f1_file = 'highest_f1_score.txt'

    if os.path.exists(param_file):
        with open(param_file, 'r') as file:
            best_params = json.load(file)
        print(f"\nLoaded best parameters from {param_file}: {best_params}")

        model = RandomForestClassifier(random_state=42, **best_params)
        f1_scorer = make_scorer(f1_score, average='weighted')
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=f1_scorer)
        mean_f1_score = scores.mean()
        best_score = mean_f1_score
        print(f'\nCross-Validation F1 Scores: {scores}')
        print(f'Mean F1 Score: {mean_f1_score}\n')

        if os.path.exists(highest_f1_file):
            with open(highest_f1_file, 'r') as file:
                highest_f1_score = float(file.read())
        else:
            highest_f1_score = 0.0
    else:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestClassifier(random_state=42)
        f1_scorer = make_scorer(f1_score, average='weighted')
        print(f"\nPerforming GridSearchCV for best parameters...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=f1_scorer, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        with open(param_file, 'w') as file:
            json.dump(best_params, file)
        print(f"\nSaved best parameters to {param_file}: {best_params}")

        best_score = grid_search.best_score_

        model = RandomForestClassifier(random_state=42, **best_params)

        print(f"\nBest GridSearch F1 Score: {best_score}\n")

        highest_f1_score = 0.0

    if best_score > highest_f1_score:
        with open(highest_f1_file, 'w') as file:
            file.write(str(best_score))
        highest_f1_score = best_score
        print(f"New highest validation F1 Score: {highest_f1_score}\n")
    else:
        print(f"Highest validation F1 Score remains: {highest_f1_score}\n")

    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test, output_file):
    """
    Makes predictions on the test data and saves the results to a CSV file.

    Parameters:
    - model (RandomForestClassifier): Trained model.
    - X_test (ndarray): Preprocessed features of the test data.
    - output_file (str): Path to the output CSV file.

    Returns:
    - None
    """
    predictions = model.predict(X_test)
    output_df = pd.DataFrame({'sii': predictions})
    output_df.to_csv(output_file, index=False)
    