"""
Author: James Kong
Course: COSI 104A - Introduction to Machine Learning
Assignment: Final Project
Date: 12/13/2024

Description:
This script serves as the main entry point for performing ensemble learning on a Kaggle competition dataset.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from data_loader import load_data, prepare_test_data
from pre_processor import preprocess_data
from model import train_model

def main():
    """
    Main function to execute the workflow:
    1. Load and preprocess the data
    2. Split the data into training and testing sets
    3. Train the model
    4. Evaluate the model
    5. Prepare and save predictions for test data
    """
    # Load and preprocess the data
    train_data, test_data, sample_submission = load_data('train.csv', 'test.csv', 'sample_submission.csv', 'series_train.parquet', 'series_test.parquet')
    X, y, common_columns, scaler, pca = preprocess_data(train_data, test_data)

    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    # Train the model
    voting_classifier = train_model(X_train, y_train)

    # Evaluate the model
    ensemble_preds = voting_classifier.predict(X_test)
    qwk_score = cohen_kappa_score(y_test, ensemble_preds, weights='quadratic')
    print(f"\nValidation Quadratic Weighted Kappa Score: {qwk_score:.4f}")

    # Prepare and save predictions for test data
    X_test_data = prepare_test_data(test_data, common_columns)
    X_test_scaled = scaler.transform(X_test_data)
    X_test_pca = pca.transform(X_test_scaled)

    ensemble_preds = voting_classifier.predict(X_test_pca)
    submission = sample_submission.copy()
    submission['sii'] = ensemble_preds
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file saved successfully.\n")
    print(submission)

if __name__ == "__main__":
    main()
