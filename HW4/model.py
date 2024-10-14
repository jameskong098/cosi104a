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
from sklearn.model_selection import cross_val_score

def train_and_evaluate_model(X_train, y_train, feature_names):
    """
    Train a DecisionTreeClassifier with cross-validation and evaluate using F1 score.

    Parameters:
    - X_train (ndarray): Preprocessed features of the training data.
    - y_train (Series): Labels of the training data.
    - feature_names (list): List of feature names.

    Returns:
    - model (DecisionTreeClassifier): Trained model.
    """
    print(f"\nTraining model with {X_train.shape[0]} samples and {X_train.shape[1]} features...\n")
    print(f"Features: {feature_names}")

    model = DecisionTreeClassifier(random_state=42)
    f1_scorer = make_scorer(f1_score)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=f1_scorer)

    print(f'\nCross-Validation F1 Scores: {scores}')
    print(f'Mean F1 Score: {scores.mean()}\n')

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