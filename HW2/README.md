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

## Explanation of Each `.py` File
### `main.py`
This is the primary script that orchestrates the entire workflow. It loads the training and test datasets, preprocesses the data, splits it into training and validation sets, and trains a separate linear regression model for each target variable. It also handles the prediction of values for both the validation and test datasets and saves the outputs and model parameters.

### `model_trainer.py`
This module contains functions related to training the linear regression models. The train_model function fits a linear regression model to the training data, while the get_model_params function extracts the model's coefficients and intercept, allowing users to understand the influence of each feature on the target variable.

### `data_loader.py`
This file is responsible for loading the datasets and preprocessing them. The preprocess_data function handles missing values and creates dummy variables for categorical features. The load_training_data and load_test_data functions read the respective CSV files and apply preprocessing.

### `utils.py`
This module includes utility functions to facilitate various tasks. The split_data function splits the dataset into training and validation sets. The save_model_params function saves the model's coefficients and intercept to a CSV file, while save_trained_model uses pickle to save the trained model for later use.

## Components Submitted
The submission includes the following components:

1. **Models**: Contains the saved linear regression models for each target column and their weights (coefficients) and bias (intercept).
2. **Validation Set Predictions**: Contains predictions for all validation sets created from the training data.
3. **Test Set Predictions**: Contains predictions made on the test set for each target column. 
4. **README.md**: This file, explaining how to run the code, load the model, and interpret the results.

## Preprocessing the Data

The `preprocess_data()` function is responsible for preparing the dataset by handling missing values and encoding categorical features as dummy variables. Here's a breakdown of the process:

### Function Overview
The `preprocess_data()` function performs two primary tasks:
1. **Handling Missing Values**: It fills missing values in both numerical and categorical columns.
2. **Creating Dummy Variables**: It converts categorical variables into a format that machine learning models can interpret.

### Steps in Preprocessing
** code from `data_loader.py`
1. **Filling Missing Values in Numerical Columns**
    ```python
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    ```
    - **What it does**: This loop identifies all numerical columns in the DataFrame (`float64` or `int64` types) and fills any missing values with the **median** of that column.
    - **Why median?**: The median is used because it is a robust statistic that is not sensitive to extreme outliers. By filling missing values with the median, the overall distribution of the data is less likely to be skewed.

2. **Filling Missing Values in Categorical Columns**
    ```python
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    ```
    - **What it does**: This loop identifies all categorical columns (i.e., columns of type `object`) and fills any missing values with the **mode** (the most frequent value) of that column.
    - **Why mode?**: Since categorical columns do not have numerical meanings, using the most frequent value (mode) is a reasonable approach to fill missing data.

3. **Creating Dummy Variables for Categorical Columns**
    ```python
    df = pd.get_dummies(df, drop_first=True)
    ```
    - **What it does**: This step converts all categorical columns into dummy variables, creating binary (0/1) columns for each category. The `drop_first=True` argument ensures that the first category is dropped to avoid multicollinearity (i.e., having redundant features).
    - **Why dummy variables?**: Machine learning models require numerical input, so categorical features need to be transformed into numerical form. Dummy variables achieve this by assigning a binary value (1 or 0) to represent each category.

### Return Value
The function returns the modified DataFrame (`df`) after processing:
- Missing values have been filled with appropriate statistics (median for numerical columns, mode for categorical columns).
- Categorical columns have been converted into dummy variables, making the data ready for modeling.

### Summary of Preprocessing
- **Numerical columns**: Missing values are replaced with the **median**.
- **Categorical columns**: Missing values are replaced with the **mode**.
- **Dummy variables**: Categorical columns are encoded as binary dummy variables, allowing machine learning algorithms to work with categorical data.

This preprocessing ensures that the data is clean, consistent, and suitable for use in machine learning models.

### Feature Selection
Custom feature sets have been defined for each target variable to enhance model performance. This approach considers the correlation between features and target variables, ensuring that more relevant features are utilized for each model.

## Splitting the Data for Validation

The script uses two methods to evaluate the performance of the linear regression models: a single train-validation split and cross-validation.

### 1. Single Train-Validation Split (`train_test_split`)

```python
from sklearn.model_selection import train_test_split

def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Splits data into training and validation sets.
    
    Args:
        df: pd.DataFrame, the dataset.
        target_col: str, the name of the target column.
        test_size: float, the proportion of the data to use as validation set.
    
    Returns:
        X_train, X_val, y_train, y_val: split features and target sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_val, y_train, y_val
```
**code from `utils.py`

### Overview:
   -  The `split_data()` function uses train_test_split to divide the dataset into training and validation sets.
   -  **Input**: The function takes the entire dataset (df), the name of the target column (target_col), and optional parameters like the test_size (proportion of data to reserve for validation) and random_state (for reproducibility).  
   -  **Output**: It returns `X_train`, `X_val`, `y_train`, and `y_val`:
   
      -  `X_train`, `y_train`: The training data (80% by default).

      -  `X_val`, `y_val`: The validation data (20% by default).

### Explanation of `X = df.drop(columns=[target_col])` and `y = df[target_col]`

### Why We Do This:

In the context of machine learning, the data is typically divided into **features** (input variables) and a **target** (the output or label we want to predict). These two lines separate the features from the target column in the dataset:

1. **`X = df.drop(columns=[target_col])`:**
   - **Purpose**: This line creates the **feature set** `X` by removing the `target_col` (the column we are trying to predict) from the dataset `df`. 
   - **Why**: We only want the input variables (features) in `X` for training the model, so we drop the target column. This ensures that the model is trained using only the independent variables.
   - **Result**: `X` contains all columns except the target column.

2. **`y = df[target_col]`:**
   - **Purpose**: This line assigns the **target variable** (the column we want to predict) to `y`. 
   - **Why**: The model needs to learn the relationship between the features and the target. By assigning the target column to `y`, we ensure that this is what the model will attempt to predict.
   - **Result**: `y` contains only the values from the target column, which will serve as the dependent variable in the model.

### In Summary:
- **`X`** (features): Contains all input variables for training the model, excluding the target column.
- **`y`** (target): Contains the output variable, which the model will learn to predict.


### Why Use `train_test_split`?

   -  This method gives a single view of the model’s performance on unseen validation data after training on a fixed portion of the dataset.
   -  By using `random_state`, the split is consistent across runs, which makes results reproducible.
   
**Example Output**:
   ```bash
   Validation R² Score (Train-Val Split with random_state = 42) for avgAnnCount (1 is Perfect): 0.7935
   ```

### 2. Cross-Validation
   ```python
   from sklearn.model_selection import cross_val_score

   # Perform cross-validation scoring
   cv_scores = cross_val_score(model, scaler.fit_transform(X_full), y_full, cv=5, scoring='r2')
   ```
   **code from `main.py`
### Overview:
   -  Cross-validation divides the dataset into k-folds (5 folds in this case), trains the model on `k-1` folds, and validates it on the remaining fold. This process is repeated `k` times, each time using a different fold for validation.
   -  **Input**: The entire dataset (`X_full` and `y_full`), the number of folds (`cv=5`), and the scoring metric (`r2`).
   -  **Output**: It returns the R² scores for each fold, providing an average performance metric across multiple train-validation splits.

### Why Use Cross-Validation?
   -  Cross-validation gives a more robust view of model performance by averaging over multiple splits, which reduces the risk of overfitting to a single validation set.
   -  It provides insight into how well the model generalizes to different subsets of the data.

```bash
Cross-Validation R² Scores for avgAnnCount: [0.73200534 0.70308607 0.94521387 0.84824397 0.80976422]
Average Cross-Validation R² Score for avgAnnCount: 0.8077
```

### Summary of Data Splitting Approaches
   -  **Train-Validation Split** (`train_test_split`): A single fixed split between training and validation sets to evaluate the model’s performance.
   -  **Cross-Validation**: A more robust approach that splits the data into multiple folds and averages the performance over these folds, providing a more reliable estimate of model performance.

## Model Training 

### How the Model is Trained

1. **Data Loading**:
   - The script begins by loading training and test datasets using the `load_training_data` and `load_test_data` functions from `data_loader.py`. 
   - The training data is preprocessed to handle missing values and create dummy variables.

2. **Target Columns and Feature Sets**:
   - The target columns are defined as `avgAnnCount`, `avgDeathsPerYear`, `TARGET_deathRate`, and `incidenceRate`.
   - Each target column has a corresponding set of relevant features specified in the `feature_sets` dictionary.

3. **Training Loop**:
   - A loop iterates over each target column:
     - The relevant features for the current target are selected.
     - The data is split into training and validation sets using the `split_data` function from `utils.py`.
     - Features are standardized using `StandardScaler`.

4. **Model Training**:
   - A linear regression model is instantiated and trained using the `train_model` function which contains the following code:
   ```python
   def train_model(X_train, y_train):
       """
       Trains a linear regression model.
      
       Args:
          X_train: pd.DataFrame, training features.
          y_train: pd.Series, training target variable.
      
       Returns:
       - model: trained linear regression model.
       """
       model = LinearRegression()
       model.fit(X_train, y_train)
      
       return model
   ```
   - The model's coefficients and intercept are retrieved using `get_model_params`.

5. **Saving Model Parameters**:
   - The coefficients and intercept are saved to a CSV file using `save_model_params`.
   - The trained model is serialized and saved as a `.pkl` file using `save_trained_model`.

6. **Validation R² Score**:
   - The model makes predictions on the validation set, and the R² score is calculated using `r2_score`. 
   - This score is printed to the console for evaluation.

7. **Test Predictions**:
   - The model generates predictions on the test dataset, which are stored for later use.

8. **Cross-Validation**:
   - Cross-validation is performed to assess the model's performance across multiple splits using `cross_val_score`, providing averaged scores.

9. **Feature Selection**:
   - Adjusting the features selected for each target column is crucial. Some features may have higher correlations with target variables than others. 
   - Start by including all features (excluding target features) in the `feature_sets`, with each target column as the key and common features as values.
   - After the initial run, record baseline R² scores and iteratively remove less relevant features to observe their impact on the scores. 
   - Consider using a correlation matrix to identify features with stronger correlations to the target variables.

## Instructions to Run the Code
1. **Prerequisites**: Ensure you have Python and the necessary libraries installed. You can install required libraries using:
   ```bash
   pip install pandas scikit-learn
2. **Data Files**: Place the data files HW1_training.csv and HW1_test_X.csv in the same directory as the script.
3. **Loading the Data:** Edit the following lines in `main.py` to include the name of your data files (HW1_training.csv and HW1_test_X.csv). This will load the data so that it can be trained:
   ```python
   train_df, target_columns, feature_sets = load_training_data("HW1_training.csv")
   test_df = load_test_data("HW1_test_X.csv")
   ```
3. **Customizing Random State**: You can customize the `random_state` parameter in the `split_data` function within `main.py` to change the data split for the validation set. Adjust it to any integer value for different random splits. Alternatively, you can also rely on the cross validation scores as it can provide a more averaged/holistic outlook.

4. **Adjusting the features selected for each target column**: Certain features have higher correlation with target features than others. Some features can be removed for certain target features as they are less related and will decrease the r-squared score. You can start initially by putting all features excluding the target features within the `data_loader.py` file within `feature_sets` with each target column acting as the key and the common features (excluding the target features) as the value. Make sure to also put the target features within the `target_columns` list. After running the script one time, you can record the baseline scores and adjust accordingly by removing features one by one to see if the score changes positively. You can try to predict which features have stronger correlation or you can create a correlation matrix to observe which features have higher correlation to the target features.

5. **Running the Script**: Execute the main script:
   ```bash
   python main.py
   ```
6. **Output**: After running the script, the following output files will be generated:
   -  `validation_predictions.csv`: Contains predictions for the validation set.
   -  `HW1_test_Y.csv`: Contains predictions for the test set.
   -  Model parameters are saved in the `trained_models/` directory along with their corresponding models.

## Interpreting the Results

When running the script, the output provides several key pieces of information for each target variable: 

### Model Parameters
For each target variable (`avgAnnCount`, `avgDeathsPerYear`, `TARGET_deathRate`, and `incidenceRate`), the model's coefficients (weights) and intercept (bias) are saved in separate `.csv` files in the following format:
- `avgAnnCount_model_params.csv`
- `avgDeathsPerYear_model_params.csv`
- `TARGET_deathRate_model_params.csv`
- `incidenceRate_model_params.csv`

These files store the influence of each feature on the target variable, helping to understand the model's decision-making process.

### Trained Model Saved
The trained models are saved as `.pkl` files in the `trained_models/` directory. Each model is named based on the target variable it predicts. For example:
- `avgAnnCount_linear_model.pkl`
- `avgDeathsPerYear_linear_model.pkl`
- `TARGET_deathRate_linear_model.pkl`
- `incidenceRate_linear_model.pkl`

These serialized models can be loaded later for making predictions on new data, allowing for reusability without retraining the model.

### Validation R² Score
The validation R² score represents how well the model performs on the validation set (a subset of the training data). This score ranges from -∞ to 1, where:
- **1**: The model perfectly predicts the target.
- **0**: The model's predictions are as good as the average of the target values.
- **Negative values**: The model performs worse than using the average of the target values.

For example:
- The validation R² score for `avgAnnCount` is **0.7935**, meaning the model explains ~79.35% of the variance in the validation set.
- In contrast, for `TARGET_deathRate`, the R² score is **0.4143**, indicating the model explains about 41.43% of the variance, suggesting room for improvement.

### Cross-Validation Scores
Cross-validation provides a more reliable estimate of model performance by splitting the data into multiple training and validation sets. The output shows the R² scores for each fold of the cross-validation, as well as the average score across all folds.

For instance:
- The average cross-validation R² score for `avgDeathsPerYear` is **0.9444**, which indicates a strong model fit.
- On the other hand, the average score for `incidenceRate` is **0.1535**, suggesting that the model struggles to explain much variance for this target.

### Summary
- **High R² Scores** (closer to 1) indicate the model is effectively capturing the relationships in the data.
- **Low or Negative R² Scores** suggest the model is either underfitting (failing to capture patterns) or requires further tuning, especially in the case of `TARGET_deathRate` and `incidenceRate`.

These results can guide further feature engineering, model adjustments, or attempts at regularization to improve performance.

## Saving the Model with Pickle
The trained linear regression model is saved using the Python `pickle` module. This allows us to serialize the model object and store it in a file for later use, enabling easy retrieval without needing to retrain the model.

### Implementation
In the `utils.py` file, the function `save_trained_model` handles the serialization process. The model is saved as a `.pkl` file in the `trained_models/` directory.

## Model Parameters
The model's coefficients and intercept for each target variable are saved in CSV files within the trained_models/ directory. Each file is named according to the target variable, e.g., avgAnnCount_model_params.csv.

## Replicating Validation and Test Results

To replicate the validation and test results and obtain the same R² scores and predictions, ensure the following conditions:

1. **Feature Selection**:
   - Use the `feature_sets` already defined in `data_loader.py` to ensure the same set of features is used for each target column. This consistency is crucial for replicating the results accurately as adding/removing different features will change the score.

2. **Set Random State**:
   - Maintain the same `random_state` value (set to 42 in the code) in the `split_data` function. This ensures that the training and validation splits are identical across different runs for the single split.


## Conclusion

This project demonstrates the application of linear regression models to predict health-related metrics based on socio-economic factors. By selecting relevant features and preprocessing the data, we aim to improve the model's performance and ensure accurate predictions.