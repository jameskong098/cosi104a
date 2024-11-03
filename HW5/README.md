# Neural Network Model

COSI 116A - Introduction to Machine Learning

HW5

James Kong

11/04/24

## Overview

This project aims to build a neural network model to predict the class labels of test samples in `test_data.csv`. The prediction results are saved in `test_label-me.csv`.

## Files

- `data_loader.py`: Contains functions to load training and test data from CSV files.
- `pre_processor.py`: Provides functions to preprocess the training and test data, including scaling numerical features and encoding categorical features.
- `model.py`: Contains functions to train and evaluate a neural network model, and to make predictions on the test data.
- `main.py`: Orchestrates the data loading, preprocessing, model training, and prediction steps.
- `test.py`: Script to load and evaluate the prediction results.
- `best_params_scores.txt`: Stores the best parameters and scores for the model.
- `train_data.csv`: Training data file.
- `test_data.csv`: Test data file.
- `test_label-me.csv`: Output file for predictions.

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

### `pre_processor.py`

- `preprocess_data(X_train, X_test)`: Preprocesses the training and test data.
  - **Parameters**:
    - `X_train` (DataFrame): Features of the training data.
    - `X_test` (DataFrame): Features of the test data.
  - **Returns**:
    - `X_train` (ndarray): Preprocessed training data.
    - `X_test` (ndarray): Preprocessed test data.

### `model.py`

- `train_nn(X_train, y_train, use_full_dataset, force_retune)`: Trains a neural network model with either cross-validation or a single train-test split.
  - **Parameters**:
    - `X_train` (ndarray): Features for training.
    - `y_train` (ndarray): Labels for training.
    - `use_full_dataset` (bool): Flag to use the full dataset for training.
    - `force_retune` (bool): Flag to force retuning of hyperparameters.
  - **Returns**:
    - `nn_model` (Pipeline): Trained neural network model.
- `save_best_params_scores(file_path, best_params, best_scores)`: Saves the best parameters and scores to a file.
  - **Parameters**:
    - `file_path` (str): Path to the file where parameters and scores will be saved.
    - `best_params` (dict): Best hyperparameters.
    - `best_scores` (dict): Best scores.
- `load_best_params_scores(file_path)`: Loads the best parameters and scores from a file.
  - **Parameters**:
    - `file_path` (str): Path to the file from which parameters and scores will be loaded.
  - **Returns**:
    - `best_params` (dict): Best hyperparameters.
    - `best_scores` (dict): Best scores.

### `test.py`

- `evaluate_predictions(predictions_file)`: Loads and evaluates the prediction results using the AUC score.
  - **Parameters**:
    - `predictions_file` (str): Path to the predictions file.
  - **Returns**:
    - `auc_score` (float): AUC score of the predictions.

### `main.py`

- `main()`: Orchestrates the data loading, preprocessing, model training, and prediction steps.
  - **Steps**:
    1. Loads the training and test data.
    2. Preprocesses the data.
    3. Trains the neural network model.
    4. Makes predictions on the test data.
    5. Saves the predictions to `test_label-me.csv`.

## How to Run

1. Ensure you have all the required libraries installed:
    ```sh
    pip install pandas scikit-learn
    ```

2. Configure the `train_nn` parameters in `main.py`:

    - `use_full_dataset` (bool): 
      - Set to `True` to use the full dataset for training with cross-validation.
      - Set to `False` to use a single train-test split for training.

    - `force_retune` (bool): 
      - Set to `True` to force retuning of hyperparameters using GridSearchCV.
      - Set to `False` to use previously saved best parameters if available.

    Example:
    ```python
    nn_model = train_nn(X_train, y_train, use_full_dataset=True, force_retune=True)
    ```

3. Run the `main.py` script to train the model and make predictions:
    ```sh
    python main.py
    ```

4. The predictions will be saved in `test_label-me.csv`.

5. To evaluate the predictions, run the `test.py` script:
    ```sh
    python test.py
    ```

## Dataset Information

- **Customer_Age**: continuous
- **Gender**: categorical
- **Dependent**: continuous
- **Education**: categorical
- **Marital**: categorical
- **Income**: categorical
- **KCategory**: categorical
- **onBook**: continuous
- **Relationship**: continuous
- **Inactive**: continuous
- **ContactNum**: continuous
- **CLimit**: continuous
- **Revolving**: continuous
- **Open**: continuous
- **Amt_41**: continuous
- **TransAmt**: continuous
- **TransCt**: continuous
- **Ct_41**: continuous
- **Utilization**: continuous
- **Importance**: continuous
- **MustHave**: continuous
- **Group**: categorical
- **Essential**: continuous
- **DropLevel**: continuous

## Submission Requirements

- **Prediction Result File**: `test_label-me.csv`
- **Explanation and Results Report**: A PDF file explaining the code and reporting the results.