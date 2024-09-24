# Linear Regression Model for Predicting Health Metrics

COSI 116A - Introduction to Machine Learning

HW2

James Kong

09/24/24

## Description
This project builds linear regression models using the training dataset `HW1_training.csv` to predict the following health metrics:
- `avgAnnCount`
- `avgDeathsPerYear`
- `TARGET_deathRate`
- `incidenceRate`

The trained models are then applied to the test dataset `HW1_test_X.csv`, and the predictions are saved in `HW1_test_Y.csv`.

## Components Submitted
The submission includes the following components:

1. **Models**: Contains the saved linear regression model and its weights (coefficients) and bias (intercept).
2. **Validation Set Predictions**: Contains predictions for a validation set created from the training data.
3. **Test Set Predictions**: Contains predictions made on the test set.
4. **README.md**: This file, explaining how to run the code, load the model, and interpret the results.

## Data Preprocessing
The data is preprocessed to handle missing values and create dummy variables for categorical features. Numerical columns are filled with their median values, and categorical columns are filled with their mode. Dummy variables are created for categorical features such as `binnedInc` and `Geography`.

### Feature Selection
Custom feature sets have been defined for each target variable to enhance model performance. This approach considers the correlation between features and target variables, ensuring that more relevant features are utilized for each model.

## Explanation of Each .py File
### `main.py`
This is the primary script that orchestrates the entire workflow. It loads the training and test datasets, preprocesses the data, splits it into training and validation sets, and trains a separate linear regression model for each target variable. It also handles the prediction of values for both the validation and test datasets and saves the outputs and model parameters.

### `model_trainer.py`
This module contains functions related to training the linear regression models. The train_model function fits a linear regression model to the training data, while the get_model_params function extracts the model's coefficients and intercept, allowing users to understand the influence of each feature on the target variable.

### `data_loader.py`
This file is responsible for loading the datasets and preprocessing them. The preprocess_data function handles missing values and creates dummy variables for categorical features. The load_training_data and load_test_data functions read the respective CSV files and apply preprocessing.

### `utils.py`
This module includes utility functions to facilitate various tasks. The split_data function splits the dataset into training and validation sets. The save_model_params function saves the model's coefficients and intercept to a CSV file, while save_trained_model uses pickle to save the trained model for later use.

## Model Parameters
The model's coefficients and intercept for each target variable are saved in CSV files within the trained_models/ directory. Each file is named according to the target variable, e.g., avgAnnCount_model_params.csv.

## Instructions to Run the Code
1. **Prerequisites**: Ensure you have Python and the necessary libraries installed. You can install required libraries using:
   ```bash
   pip install pandas scikit-learn
2. **Data Files**: Place the data files HW1_training.csv and HW1_test_X.csv in the same directory as the script.

3. **Running the Script**: Execute the main script:
   ```bash
   python main.py

4. **Customizing Random State**: You can customize the `random_state` parameter in the `split_data` function within `main.py` to change the data split for the validation set. Adjust it to any integer value for different random splits.

5. **Output**: After running the script, the following output files will be generated:
   -  `validation_predictions.csv`: Contains predictions for the validation set.
   -  `HW1_test_Y.csv`: Contains predictions for the test set.
   -  Model parameters are saved in the `trained_models/` directory.

## Saving the Model with Pickle
The trained linear regression model is saved using the Python `pickle` module. This allows us to serialize the model object and store it in a file for later use, enabling easy retrieval without needing to retrain the model.

### Implementation
In the `utils.py` file, the function `save_trained_model` handles the serialization process. The model is saved as a `.pkl` file in the `trained_models/` directory. The following steps are performed:

## Model Parameters
The model's coefficients and intercept for each target variable are saved in CSV files within the trained_models/ directory. Each file is named according to the target variable, e.g., avgAnnCount_model_params.csv.

## Conclusion

This project demonstrates the application of linear regression models to predict health-related metrics based on socio-economic factors. By selecting relevant features and preprocessing the data, we aim to improve the model's performance and ensure accurate predictions.