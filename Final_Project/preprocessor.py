from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

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
    # Ensure both datasets have the same columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

    # Fill missing values in categorical columns with 'missing'
    X_train[categorical_cols] = X_train[categorical_cols].fillna('missing')
    X_test[categorical_cols] = X_test[categorical_cols].fillna('missing')

    # Fill missing values in numerical columns with the mean
    X_train[numerical_cols] = X_train[numerical_cols].fillna(X_train[numerical_cols].mean())
    X_test[numerical_cols] = X_test[numerical_cols].fillna(X_train[numerical_cols].mean())

    # Ensure all columns have consistent data types
    X_train[categorical_cols] = X_train[categorical_cols].astype(str)
    X_test[categorical_cols] = X_test[categorical_cols].astype(str)
    X_train[numerical_cols] = X_train[numerical_cols].astype(float)
    X_test[numerical_cols] = X_test[numerical_cols].astype(float)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Fit the preprocessor and transform the data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Extract feature names
    num_features = numerical_cols
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    feature_names = num_features + cat_features

    return X_train, X_test, feature_names
