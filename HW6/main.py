"""
Author: James Kong
Course: COSI 116A - Introduction to Machine Learning
Assignment: HW6
Date: 11/12/24

Description:
This script serves as the main entry point for training an ensemble model and making predictions on test data.
It first imports the necessary functions from the `data_loader` and `model` modules. The `load_data` function is used to load training and testing datasets from specified CSV files. 
The `train_ensemble` function trains an ensemble model using the training data, with an option to perform cross-validation. 
After training, the model predicts the labels for the test dataset, and the predictions are saved to a CSV file.
"""
from data_loader import load_data
from model import train_ensemble
from pre_processor import preprocess_data
import pandas as pd

def main():
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    output_file = "test_label.csv"
    
    X_train, y_train, X_test = load_data(train_file, test_file)

    X_train, X_test = preprocess_data(X_train, X_test)
    
    # Set use_full_dataset to False to use single train-test split
    ensemble_model = train_ensemble(X_train, y_train, use_full_dataset=True)
    
    predictions = ensemble_model.predict(X_test)
    
    pd.DataFrame(predictions, columns=["Class"]).to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
