"""
model.py

This module contains functions to train and evaluate a machine learning model.
It includes functions to train the model using cross-validation and to make predictions.

Functions:
- select_best_algorithm(X_train, y_train, run=False): Selects the best algorithm based on cross-validation.
- train_and_evaluate_model(X_train, y_train, feature_names): Trains and evaluates the model.
- make_predictions(model, X_test, output_file, test_ids): Makes predictions on the test data and saves the results.
"""

import pandas as pd
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def train_and_evaluate_model(X_train, y_train, feature_names):
    """
    Train multiple models using cross-validation and evaluate using F1 score and Kappa score.
    Save the highest validation F1 score to highest_f1_score.txt and update it if the latest score is higher.

    Parameters:
    - X_train (ndarray): Preprocessed features of the training data.
    - y_train (Series): Labels of the training data.
    - feature_names (list): List of feature names.

    Returns:
    - model (dict): Dictionary of trained models.
    """
    print(f"\nTraining model with {X_train.shape[0]} samples and {X_train.shape[1]} features...\n")
    print(f"Features: {feature_names}")

    models = {
        'XGBoost': XGBClassifier(random_state=42, objective='multi:softmax', num_class=4),
        'LightGBM': LGBMClassifier(random_state=42, objective='multiclass', num_class=4),
        'TabNet': TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='sparsemax',
            verbose=0
        ),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0, loss_function='MultiClass')
    }

    best_score = 0
    best_model = None

    kappa_scorer = make_scorer(cohen_kappa_score)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == 'TabNet':
            model.fit(X_train, y_train, patience=30, max_epochs=1000, eval_metric=['logloss'])
        else:
            model.fit(X_train, y_train)
        
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=kappa_scorer)
        mean_kappa_score = scores.mean()
        print(f'{name} Cross-Validation Cohen Kappa Score: {mean_kappa_score}')

        if mean_kappa_score > best_score:
            best_score = mean_kappa_score
            best_model = model

    print(f'\nBest Model: {best_model.__class__.__name__} with Cohen Kappa Score: {best_score}\n')

    return best_model, models

def make_predictions(models, X_test, output_file, test_ids):
    """
    Makes predictions on the test data using ensemble learning and saves the results to a CSV file.

    Parameters:
    - models (dict): Dictionary of trained models.
    - X_test (ndarray): Preprocessed features of the test data.
    - output_file (str): Path to the output CSV file.
    - test_ids (Series): IDs of the test data.

    Returns:
    - None
    """
    predictions = []
    for name, model in models.items():
        print(f"Making predictions with {name}...")
        if name == 'TabNet':
            preds = model.predict(X_test)
        else:
            preds = model.predict(X_test)
        predictions.append(preds)

    # Ensemble predictions by averaging
    final_predictions = sum(predictions) / len(predictions)
    final_predictions = final_predictions.argmax(axis=1)  # Get the class with the highest probability

    submission_df = pd.DataFrame({
        'id': test_ids,
        'sii': final_predictions
    })

    submission_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    