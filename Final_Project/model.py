from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    lgbm_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    voting_classifier = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ]
    )

    voting_classifier.fit(X_train, y_train)
    return voting_classifier

def quadratic_weighted_kappa(y_true, y_pred, num_ratings=None):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    if num_ratings is None:
        num_ratings = max(max(y_true), max(y_pred)) + 1

    O = np.zeros((num_ratings, num_ratings))
    for a, p in zip(y_true, y_pred):
        O[a, p] += 1

    actual_hist = np.sum(O, axis=1)
    predicted_hist = np.sum(O, axis=0)
    E = np.outer(actual_hist, predicted_hist) / np.sum(O)

    W = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    numerator = np.sum(W * O)
    denominator = np.sum(W * E)
    kappa = 1 - numerator / denominator

    return kappa
