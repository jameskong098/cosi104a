# Kaggle Project: Predicting Problematic Internet Use in Children

**Author:** James Kong

**Course:** COSI 104A - Introduction to Machine Learning

**Assignment:** Final Project

**Date:** 12/13/2024

## Overview

In today’s digital age, problematic internet use among children and adolescents is a growing concern. As part of my project for the COSI 104A - Introduction to Machine Learning course, I aimed to develop a predictive model to identify early indicators of problematic internet use based on children's physical activity data. This project leverages accessible physical and fitness measures as proxies for detecting problematic internet use, which can help in contexts lacking clinical expertise or suitable assessment tools. The goal is to enable prompt interventions aimed at promoting healthier digital habits among children.

## Project Structure

The project is organized as follows:

- `data_loader.py`: Contains functions to load and prepare the data from CSV and Parquet files.
  - `load_data`: Loads and merges training and test data from CSV and Parquet files.
  - `prepare_test_data`: Prepares test data for prediction by mapping season strings to integers and filling missing values.
- `pre_processor.py`: Includes functions for preprocessing the data, such as scaling and applying PCA.
  - `preprocess_data`: Preprocesses the training and test data by removing columns with missing values, mapping season strings to integers, encoding categorical columns, and applying PCA.
- `model.py`: Defines the function to train a voting classifier model using RandomForest, XGBoost, and LightGBM classifiers.
  - `train_model`: Trains a voting classifier model using RandomForest, XGBoost, and LightGBM classifiers.
- `main.py`: The main script to execute the workflow, including loading data, preprocessing, training the model, evaluating, and saving predictions.
  - `main`: Executes the workflow by loading and preprocessing data, splitting data into training and testing sets, training the model, evaluating the model, and preparing and saving predictions.
- `plot_performance.py`: Script to plot the performance of the model over multiple rounds of submissions.
- `README.md`: Project documentation.
- `data_dictionary.csv`: Contains descriptions of the dataset fields.
- `train.csv`, `test.csv`, `sample_submission.csv`: Dataset files.
- `series_train.parquet/`, `series_test.parquet/`: Additional dataset files in Parquet format.
## Feature Engineering

Feature engineering is handled within the `preprocess_data` function in the [`pre_processor.py`](pre_processor.py) file. The steps include:

1. Removing columns with more than 50% missing values:
    ```python
    train_data = train_data.loc[:, train_data.isnull().mean() < 0.5]
    ```

2. Filling remaining missing values with 0:
    ```python
    train_data = train_data.fillna(0)
    ```

3. Mapping season strings to integers for relevant columns:
    ```python
    season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    season_cols = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 'PCIAT-Season', 'SDS-Season', 'PreInt_EduHx-Season']
    for col in season_cols:
        if col in train_data.columns:
            train_data[col] = train_data[col].map(season_mapping).fillna(-1).astype(int)
    ```

4. Converting all columns to numeric:
    ```python
    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    ```

5. Selecting columns with high correlation to the label `sii`:
    ```python
    correlation_with_label = train_data.corr()['sii']
    corr_threshold = 0.2
    high_corr_columns = correlation_with_label[abs(correlation_with_label) >= corr_threshold].index
    train_data = train_data[high_corr_columns]
    ```

6. Encoding categorical columns using [LabelEncoder](http://_vscodecontentref_/0):
    ```python
    encoder = LabelEncoder()
    categorical_columns = train_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        train_data[col] = encoder.fit_transform(train_data[col].astype(str))
    ```

7. Standardizing the features using [StandardScaler](http://_vscodecontentref_/1):
    ```python
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_data)
    ```

8. Applying PCA to reduce dimensionality:
    ```python
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    ```

## Algorithm Selection

Initially, I started with a RandomForest classifier but found more success with ensemble learning as I tested multiple models and evaluated their scores. The `train_model` function in the [`model.py`](model.py) file defines the training process for a voting classifier model using RandomForest, XGBoost, and LightGBM classifiers. The ensemble approach improved the model's performance by combining the strengths of different algorithms.

## Performance Plot

![Performance Plot](performance_chart.png)

### Evaluation Metric

The `cohen_kappa_score` from `sklearn.metrics` is used to evaluate the model's performance. The Cohen's Kappa score measures the agreement between two raters who each classify N items into C mutually exclusive categories. The score ranges from -1 to 1, where:

- 1 indicates perfect agreement.
- 0 indicates no agreement (random chance).
- Negative values indicate disagreement.

In this project, the `cohen_kappa_score` is used to compare the predicted labels with the true labels. The score is calculated as follows:

```python
from sklearn.metrics import cohen_kappa_score

# Example usage
qwk_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
```

A score of **0.189 - 0.32** indicates moderate agreement. While it shows some predictive power, there is room for improvement. In the context of this project, these scores suggest that the model is able to capture some patterns in the data but would benefit from further tuning and feature engineering.

## How to Run

1. **Install Dependencies**: Ensure you have all the required Python packages installed. You can use the following commands to install the necessary packages:

    Using `pip`:
    ```sh
    pip install pandas scikit-learn xgboost lightgbm matplotlib tqdm
    ```

    Using `conda`:
    ```sh
    conda install pandas scikit-learn xgboost lightgbm matplotlib tqdm
    ```

2. **Execute the Main Script**: Run the [main.py](http://_vscodecontentref_/0) script to perform the entire workflow:
    ```sh
    python main.py
    ```

## Conclusion

This project demonstrates the use of machine learning techniques to predict problematic internet use in children based on physical activity data. By leveraging accessible physical and fitness measures, the model aims to provide early indicators of problematic internet use, enabling timely interventions to promote healthier digital habits.

## Acknowledgements

[Kaggle - johnsonhk88](https://www.kaggle.com/code/johnsonhk88/cmi-problematic-internet-use-ml)