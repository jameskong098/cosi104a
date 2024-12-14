"""
model.py

This script defines the function to train a voting classifier model using RandomForest, XGBoost, and LightGBM classifiers.
"""

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_model(X_train, y_train):
    """
    Train a voting classifier model using RandomForest, XGBoost, and LightGBM classifiers.

    Parameters:
    X_train (array-like): Training data features
    y_train (array-like): Training data labels

    Returns:
    VotingClassifier: Trained voting classifier model
    """
    # Initialize individual models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    lgbm_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    # Create a voting classifier with the individual models
    voting_classifier = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ]
    )

    # Train the voting classifier
    voting_classifier.fit(X_train, y_train)

    return voting_classifier
