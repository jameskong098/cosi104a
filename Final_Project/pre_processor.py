"""
pre_processor.py

This script defines the function to preprocess the training and test data.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(train_data, test_data):
    """
    Preprocess the training and test data.

    Parameters:
    train_data (DataFrame): Training data
    test_data (DataFrame): Test data

    Returns:
    tuple: Processed features, labels, common columns, scaler, and PCA object
    """
    # Remove columns with more than 50% missing values
    threshold = 0.5 * len(train_data)
    columns_with_data = train_data.columns[train_data.isnull().sum() < threshold]
    train_data = train_data[columns_with_data]
    
    # Fill remaining missing values with 0
    train_data.fillna(0, inplace=True)
    
    # Drop rows where 'sii' is missing
    train_data.dropna(subset=['sii'], inplace=True)

    # Map season strings to integers
    season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    season_cols = [
        'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season',
        'FGC-Season', 'BIA-Season', 'PCIAT-Season', 'SDS-Season', 'PreInt_EduHx-Season'
    ]

    for col in season_cols:
        if col in train_data.columns:
            train_data[col] = train_data[col].map(season_mapping).fillna(-1).astype(int)

    # Convert all columns to numeric
    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    
    # Select columns with high correlation to the label 'sii'
    correlation_with_label = train_data.corr()['sii']
    corr_threshold = 0.2
    high_corr_columns = correlation_with_label[abs(correlation_with_label) >= corr_threshold].index
    train_data = train_data[high_corr_columns]

    # Find common columns between train and test data
    common_columns = train_data.columns.intersection(test_data.columns)
    X = train_data[common_columns]
    y = train_data['sii']

    # Encode categorical columns
    encoder = LabelEncoder()
    categorical_columns = X.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        X[col] = encoder.fit_transform(X[col].astype(str))

    # Drop 'id' column if present
    X.drop(columns=['id'], errors='ignore', inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, common_columns, scaler, pca
