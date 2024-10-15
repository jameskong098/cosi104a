# Fraud Detection Decision Tree Model

COSI 116A - Introduction to Machine Learning

HW4

James Kong

10/17/24

## Overview

This project aims to build a model for detecting fraudulent transactions using `DecisionTreeClassifier`. The model is trained using the provided training data and then used to make predictions on the test data. The predictions are saved in a CSV file.

## Files

- `data_loader.py`: Contains functions to load training and test data from CSV files.
- `preprocessor.py`: Provides functions to preprocess the training and test data, including scaling numerical features and encoding categorical features.
- `model.py`: Contains functions to train and evaluate a machine learning model, and to make predictions on the test data.
- `main.py`: Orchestrates the data loading, preprocessing, model training, and prediction steps.
- `timer.py`: Utility functions to measure the time taken for various steps.
- `highest_f1_score.txt`: Stores the highest validation F1 score achieved.
- `hyper_parameters.txt`: Stores the best hyperparameters found during model training.
- `HW4_training.csv`: Training data file.
- `HW4_test_input.csv`: Test data file.
- `HW4_test_output.csv`: Output file for predictions.

## Functions

### `data_loader.py`

- `load_data(train_path, test_path)`: Loads training and test data from CSV files.
  - **Parameters**:
    - `train_path` (str): Path to the training data CSV file.
    - `test_path` (str): Path to the test data CSV file.
  - **Returns**:
    - `X_train` (DataFrame): Features of the training data.
    - `y_train` (Series): Labels of the training data.
    - `X_test` (DataFrame): Features of the test data.

### `preprocessor.py`

- `preprocess_data(X_train, X_test)`: Preprocesses the training and test data.
  - **Parameters**:
    - `X_train` (DataFrame): Features of the training data.
    - `X_test` (DataFrame): Features of the test data.
  - **Returns**:
    - `X_train` (ndarray): Preprocessed training data.
    - `X_test` (ndarray): Preprocessed test data.
    - `feature_names` (list): List of feature names.

### `model.py`

- `train_and_evaluate_model(X_train, y_train, feature_names)`: Trains and evaluates the model.
  - **Parameters**:
    - `X_train` (ndarray): Preprocessed features of the training data.
    - `y_train` (Series): Labels of the training data.
    - `feature_names` (list): List of feature names.
  - **Returns**:
    - `model` (DecisionTreeClassifier): Trained model.

- `make_predictions(model, X_test, output_file)`: Makes predictions on the test data and saves the results.
  - **Parameters**:
    - `model` (DecisionTreeClassifier): Trained model.
    - `X_test` (ndarray): Preprocessed features of the test data.
    - `output_file` (str): Path to the output CSV file.
  - **Returns**:
    - None

### `main.py`

- `main()`: Orchestrates the data loading, preprocessing, model training, and prediction steps.
  - **Steps**:
    1. Starts the timer.
    2. Loads the training and test data.
    3. Preprocesses the data.
    4. Trains and evaluates the model.
    5. Makes predictions on the test data.
    6. Saves the predictions to `HW4_test_output.csv`.
    7. Prints the time taken for the entire process.

### `timer.py`

- `start_timer()`: Starts the timer.
- `get_time_passed(start_time)`: Prints the time passed since the timer started.

## How to Run

1. Place your training and testing CSV files in the same directory as the script. Ensure that the training file contains a `isFraud` column.

2. Install the necessary Python packages:
    ```sh
    pip install pandas scikit-learn
    ```

3. The script retrieves hyperparameters from the `hyper_parameters.txt` file. If this file does not exist, the script will perform hyperparameter tuning and generate a new `hyper_parameters.txt` file with the best parameters and use them on the current and future runs. If you want to regenerate hyperparameters and tune them again, delete the existing `hyper_parameters.txt` file before running the script. This will take a longer time than usual depending on how many cpu cores your computer has.

4. The `highest_f1_score.txt` file keeps track of the highest validation F1 score achieved during optimization. If you want to reset this score, delete the `highest_f1_score.txt` file.

5. Run the `main.py` script:
    ```sh
    python main.py
    ```

6. The predictions will be saved in `HW4_test_output.csv`.
