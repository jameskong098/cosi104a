"""
preprocessor.py

This module provides functions to preprocess the training and test data.
It includes functions to scale numerical features and encode categorical features.

Functions:
- preprocess_data(X_train, X_test): Preprocesses the training and test data.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(X_train, X_test):
    """
    Preprocess the training and test data.

    Parameters:
    - X_train (DataFrame): Features of the training data.
    - X_test (DataFrame): Features of the test data.

    Returns:
    - X_train (ndarray): Preprocessed training data.
    - X_test (ndarray): Preprocessed test data.
    - feature_names (list): List of feature names.
    """
    categorical_cols = ['Gender', 'Education', 'Marital', 'Income', 'KCategory']
    numerical_cols = X_train.columns.difference(categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    # Fit the preprocessor and transform the data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test
