"""
model.py

This module contains functions to train a machine learning model and make predictions.
It uses a DecisionTreeClassifier and evaluates the model using cross-validation with F1 score.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

def train_and_evaluate_model(X_train, y_train):
    """
    Train a DecisionTreeClassifier with cross-validation and evaluate using F1 score.

    Parameters:
    - X_train (ndarray): Preprocessed features of the training data.
    - y_train (Series): Labels of the training data.

    Returns:
    - model (DecisionTreeClassifier): Trained model.
    """
    model = DecisionTreeClassifier(random_state=42)
    f1_scorer = make_scorer(f1_score)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=f1_scorer)

    print(f'\nCross-Validation F1 Scores: {scores}')
    print(f'Mean F1 Score: {scores.mean()}\n')

    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test, output_path):
    """
    Make predictions on the test data and save to a CSV file.

    Parameters:
    - model (DecisionTreeClassifier): Trained model.
    - X_test (ndarray): Preprocessed features of the test data.
    - output_path (str): Path to save the predictions CSV file.
    """
    predictions = model.predict(X_test)
    output = pd.DataFrame(predictions, columns=['isFraud'])
    output.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}\n")
    