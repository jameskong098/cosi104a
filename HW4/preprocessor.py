"""
preprocessor.py

This module contains functions to preprocess the training and test data.
It handles scaling of numerical features and encoding of categorical features.
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(X_train, X_test):
    """
    Preprocess the training and test data.

    Parameters:
    - X_train (DataFrame): Features of the training data.
    - X_test (DataFrame): Features of the test data.

    Returns:
    - X_train (ndarray): Preprocessed training data.
    - X_test (ndarray): Preprocessed test data.
    """
    categorical_cols = ['type']
    numerical_cols = X_train.columns.difference(categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test
