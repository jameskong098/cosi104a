"""
Author: James Kong
Course: COSI 116A - Introduction to Machine Learning
Assignment: HW3
Date: 10/08/24

Description:
This script serves as the main entry point for training a Support Vector Machine (SVM) model and making predictions on test data.
It first imports the necessary functions from the `data_loader` and `model` modules. The `load_data` function is used to load training and testing datasets from specified CSV files. 
The `train_svm` function trains an SVM model using the training data, with an option to perform cross-validation. 
After training, the model predicts the labels for the test dataset, and the predictions are saved to a CSV file.
"""
from data_loader import load_data
from model import train_svm
import pandas as pd

def main():
    train_file = "HW3_Train.csv"
    test_file = "HW3_Test.csv"
    output_file = "HW3_Test_Output.csv"
    
    X_train, y_train, X_test = load_data(train_file, test_file)
    
    # Set use_cross_val to False to use single train-test split
    svm_model = train_svm(X_train, y_train, use_cross_val=False)
    
    predictions = svm_model.predict(X_test)
    
    pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
