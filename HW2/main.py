"""
Author: James Kong
Course: COSI 116A - Introduction to Machine Learning
Assignment: HW2
Date: 09/24/24

Description:
This script builds linear regression models using HW1_training.csv to predict avgAnnCount, avgDeathsPerYear, 
TARGET_deathRate, and incidenceRate. Predictions are made on HW1_test_X.csv and saved in HW1_test_Y.csv. 
Components submitted include model weights, validation predictions, test predictions, and a README.md for instructions.
"""

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from data_loader import load_training_data, load_test_data
from utils import split_data, save_model_params, save_trained_model
from model_trainer import train_model, get_model_params

def main():
    train_df, target_columns, feature_sets = load_training_data("HW1_training.csv")
    test_df = load_test_data("HW1_test_X.csv")
    
    validation_single_split_predictions = pd.DataFrame()
    test_predictions = pd.DataFrame()

    # Train a separate model for each target column and generate predictions
    for target_col in target_columns:
        print(f"\nTraining model for {target_col}...")

        relevant_features = feature_sets[target_col]
        
        # Train-validation split using a custom random_state
        random_state = 42
        X_train, X_val, y_train, y_val = split_data(train_df[relevant_features + [target_col]], target_col=target_col, random_state=random_state)
        
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = train_model(X_train_scaled, y_train)
        
        coefficients, intercept = get_model_params(model)
        print(f"\nCoefficients for {target_col}:", coefficients)
        print(f"Intercept for {target_col}:", intercept, "\n")

        # Save model coefficients and intercept to .csv file in trained_models directory
        save_model_params(target_col, coefficients, intercept)

        # Save the trained model as a .pkl file to trained_models directory
        save_trained_model(model, target_col)

        # Validation R² Score for single split with custom random_state
        y_val_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, y_val_pred)
        print(f"Validation R² Score (Train-Val Split with random_state = {random_state}) for {target_col} (1 is Perfect): {r2:.4f}\n")
        
        validation_single_split_predictions[target_col] = y_val_pred

        # Generate predictions based on test data 
        X_test = test_df[relevant_features]
        X_test_scaled = scaler.transform(X_test)
        y_test_pred = model.predict(X_test_scaled)
        
        test_predictions[target_col] = y_test_pred

        # Perform cross-validation scoring to get averaged scores from multiple splits
        X_full = train_df[relevant_features]
        y_full = train_df[target_col]
        cv_scores = cross_val_score(model, scaler.fit_transform(X_full), y_full, cv=5, scoring='r2')
        print(f"Cross-Validation R² Scores for {target_col}: {cv_scores}")
        print(f"Average Cross-Validation R² Score for {target_col}: {cv_scores.mean():.4f}\n")
        print("===================\n")
    
    validation_single_split_predictions.to_csv('validation_predictions.csv', index=False)
    print("Validation predictions saved to validation_predictions.csv.\n")

    # Ensure correct order for target columns
    test_predictions = test_predictions[target_columns]
    test_predictions.to_csv('HW1_test_Y.csv', index=False)
    print("Test predictions saved to HW1_test_Y.csv.\n")

if __name__ == "__main__":
    main()
