# Support Vector Machine Classifier

COSI 116A - Introduction to Machine Learning

HW3

James Kong

10/08/24

## Overview

This project involves training a Support Vector Machine (SVM) model using the provided training dataset and making predictions on a testing dataset. The main script orchestrates the loading of data, model training, and prediction saving. The project is structured into three main Python files:

- `data_loader.py`: Contains functionality to load the training and testing data from CSV files.
- `main.py`: The main entry point for running the SVM model training and prediction.
- `model.py`: Contains the implementation of the SVM model training with optional cross-validation.

## File Descriptions

### data_loader.py

This module includes the function to load the datasets.

- **Function: `load_data(train_file, test_file)`**
  
  Loads training and testing data from specified CSV files.

  **Parameters:**
  - `train_file` (str): Path to the CSV file containing the training data, including labels.
  - `test_file` (str): Path to the CSV file containing the testing data.

  **Returns:**
  - `X_train` (DataFrame): Features for the training set.
  - `y_train` (Series): Labels for the training set.
  - `X_test` (DataFrame): Features for the testing set.

### main.py

The main script that executes the SVM model training and predictions.

- **Function: `main()`**

  This function serves as the entry point for the program. It loads the data, trains the SVM model, and saves the predictions.

### model.py

This module contains the SVM model training logic.

- **Function: `train_svm(X_train, y_train, use_cross_val=True)`**
  
  Trains the SVM model with the option to perform cross-validation or a single train-test split.

  **Parameters:**
  - `X_train`: Features for training.
  - `y_train`: Labels for training.
  - `use_cross_val` (bool): Whether to use cross-validation (default is `True`).

  **Returns:**
  - Trained SVM model.

## How to Run the Code

1. Ensure you have the required libraries installed. You can install them using:
   ```bash
   pip install pandas scikit-learn
   ```

2. Place your training and testing CSV files in the same directory as the script. Ensure that the training file contains a `Label` column.

3. Run the `main.py` script:
   ```bash
   python main.py
   ```

4. The predictions will be saved to a file named `HW3_Test_Output.csv`.
