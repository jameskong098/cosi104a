# Ensemble Model

COSI 116A - Introduction to Machine Learning

HW6

James Kong

11/12/24

## Overview
This project aims to build an ensemble model to predict the `Class` labels of test samples in `test_data.csv`. The prediction results are saved in `test_label.csv`.

## Files
- `data_loader.py`: Contains functions to load training and test data from CSV files.
- `pre_processor.py`: Provides functions to preprocess the training and test data, including scaling numerical features and encoding categorical features.
- `model.py`: Contains functions to train and evaluate an ensemble model, and to make predictions on the test data.
- `main.py`: Orchestrates the data loading, preprocessing, model training, and prediction steps.
- `test.py`: Script to load and evaluate the prediction results.
- `train_data.csv`: Training data file.
- `test_data.csv`: Test data file.
- `test_label.csv`: Output file for predictions.

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
- `train_ensemble(X_train, y_train, use_full_dataset=True)`: Trains an ensemble model using the training data.
  - **Parameters**:
    - `X_train` (ndarray): Preprocessed features of the training data.
    - `y_train` (Series): Labels of the training data.
    - `use_full_dataset` (bool): Whether to use the full dataset for training (and using cross-validation) or partial dataset (and using train_test_split).
  - **Returns**:
    - `model` (object): Trained ensemble model.

### `main.py`
- `main()`: Orchestrates the data loading, preprocessing, model training, and prediction steps.
  - **Steps**:
    1. Loads the training and test data using `load_data`.
    2. Preprocesses the data using `preprocess_data`.
    3. Trains the ensemble model using `train_ensemble`.
    4. Makes predictions on the test data.
    5. Saves the predictions to `test_label.csv`.

### `test.py`
- `compute_f1_score(df)`: Computes the F1 score by comparing the class labels with themselves.
  - **Parameters**:
    - `df` (pd.DataFrame): DataFrame containing the class labels in the 'Class' column.
  - **Returns**:
    - `float`: F1 score (comparing the column with itself).

## Model Selection
The model selection process involved evaluating different machine learning algorithms and selecting the best-performing model based on cross-validation scores. The chosen model is an ensemble approach using a Bagging Classifier, which combines multiple base estimators to improve the overall performance and robustness.

## Ensemble Approach
The ensemble model is trained using a Bagging Classifier with 50 base estimators. Cross-validation is used to evaluate the model's performance, and the final model is trained on the full training dataset. This approach helps to reduce overfitting and improve generalization.

## Results
### Training Results
```bash
Cross-validation F1 scores: [0.81968845 0.82938026 0.84650142 0.82246247 0.84589185]
Mean cross-validation F1 Score: 0.8327848907637984
```

### Interpretation of Results
The `F1 score` is a performance metric for classification models. It measures the balance between precision and recall. An F1 score of 1.0 indicates perfect precision and recall, while a score of 0 indicates the worst performance. In this project, the mean cross-validation F1 score is approximately 0.8328. This score indicates that the ensemble model has a good balance between precision and recall in the training data. Generally, an F1 score above 0.85 is considered good, and a score above 0.9 is considered excellent. Therefore, our model's performance is satisfactory since it is very close to 0.85.

### Prediction on Test Data
The trained ensemble model is used to predict the "Class" labels for the test samples in `test_data.csv`. The predicted labels are saved in the `test_label.csv` file. The F1 score provides confidence that the model will perform well on unseen test data, making accurate predictions for the "Class" labels. The `test_label.csv` file contains the predicted "Class" labels, which can be used for further analysis or evaluation.

## Conclusion
The ensemble model trained in this project demonstrates good performance with a mean cross-validation F1 score of 0.8328. This indicates that the model is most likely effective in predicting the "Class" labels for the given dataset.