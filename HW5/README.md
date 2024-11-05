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
    - `feature_names` (list): List of feature names.

### `model.py`

- `train_nn(X_train, y_train, use_full_dataset=True)`: Trains a neural network model using the training data.
  - **Parameters**:
    - `X_train` (ndarray): Preprocessed features of the training data.
    - `y_train` (Series): Labels of the training data.
    - `use_full_dataset` (bool): Whether to use the full dataset for training (and using cross validation) or partial dataset and (and using train_test_split).
  - **Returns**:
    - `model` (object): Trained neural network model.

### `main.py`

- `main()`: Orchestrates the data loading, preprocessing, model training, and prediction steps.
  - **Steps**:
    1. Loads the training and test data using `load_data`.
    2. Preprocesses the data using `preprocess_data`.
    3. Trains the neural network model using `train_nn`.
    4. Makes predictions on the test data.
    5. Saves the predictions to `test_label-me.csv`.

### `test.py`

- `evaluate_predictions(predictions_file, true_labels_file)`: Evaluates the prediction results.
  - **Parameters**:
    - `predictions_file` (str): Path to the CSV file containing the predicted labels.
    - `true_labels_file` (str): Path to the CSV file containing the true labels.
  - **Returns**:
    - `accuracy` (float): Accuracy of the predictions.

## Results

### Training Results
```bash
Cross-validation AUC scores: [0.89212793, 0.89845066, 0.90024429, 0.88768563, 0.88260366]

Mean cross-validation AUC: 0.8922224365155899
```
### Interpretation of Results

The `AUC (Area Under the ROC Curve)` score is a performance metric for classification models. It measures the ability of the model to distinguish between classes. An AUC score of 0.5 indicates no discrimination (i.e., random guessing), while an AUC score of 1.0 indicates perfect discrimination.

In this project, the mean cross-validation AUC score is approximately 0.8922. This score indicates that the neural network model has a good ability to distinguish between the classes in the training data. Generally, an AUC score above 0.85 is considered good, and a score above 0.9 is considered excellent. Therefore, our model's performance is quite satisfactory.

### Prediction on Test Data

The trained neural network model is used to predict the "Class" labels for the test samples in `test_data.csv`. The predicted labels are saved in the `test_label-me.csv` file. The AUC score provides confidence that the model will perform well on unseen test data, making accurate predictions for the "Class" labels.

The `test_label-me.csv` file contains the predicted "Class" labels, which can be used for further analysis or evaluation.

## Conclusion

The neural network model trained in this project demonstrates good performance with a mean cross-validation AUC score of 0.8922. This indicates that the model is most likely effective in predicting the "Class" labels for the given dataset.