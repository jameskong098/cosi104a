"""
data_loader.py

This script defines functions to load and preprocess the training and test data from CSV and parquet files.
"""

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

def loadParquetFile(directory, fileName):
    """
    Read parquet file.

    Parameters:
    directory (str): Directory containing the parquet files
    fileName (str): Name of the parquet file

    Returns:
    tuple: Statistics of the parquet file and the file ID
    """
    path = os.path.join(directory, fileName, "part-0.parquet")
    df = pd.read_parquet(path)
    df.drop("step", axis=1, inplace=True)  
    statDF = df.describe().values.reshape(-1)
    return statDF, fileName.split("=")[1]  

def loadTimeSeriesData(directory):
    """
    Load time series data from parquet files.

    Parameters:
    directory (str): Directory containing the parquet files

    Returns:
    DataFrame: DataFrame containing the statistics of the parquet files
    """
    filesIds = os.listdir(directory) 
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: loadParquetFile(directory, fname), filesIds), total=len(filesIds)))
    statistic, ids = zip(*results)  
    data = pd.DataFrame(statistic, columns=[f"stat_{i}" for i in range(len(statistic[0]))])
    data["id"] = ids  
    return data

def load_data(train_dir, test_dir, sample_dir, train_parquet_dir, test_parquet_dir):
    """
    Load and merge training and test data from CSV and parquet files.

    Returns:
    tuple: Training data, test data, and sample submission DataFrame
    """
    # Load CSV files
    train_data = pd.read_csv(train_dir)
    test_data = pd.read_csv(test_dir)
    sample_submission = pd.read_csv(sample_dir)
    
    # Drop rows where 'sii' is missing
    train_data = train_data.dropna(subset=['sii'])

    # Load parquet files
    train_parquet_data = loadTimeSeriesData(train_parquet_dir)
    test_parquet_data = loadTimeSeriesData(test_parquet_dir)

    # Merge CSV and parquet data
    train_data = pd.merge(train_data, train_parquet_data, how="left", on='id')
    test_data = pd.merge(test_data, test_parquet_data, how="left", on='id')

    return train_data, test_data, sample_submission

def prepare_test_data(test_data, common_columns):
    """
    Prepare test data for prediction.

    Parameters:
    test_data (DataFrame): Test data
    common_columns (Index): Columns common to both training and test data

    Returns:
    DataFrame: Processed test data
    """
    # Map season strings to integers
    season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    season_cols = [
        'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season',
        'FGC-Season', 'BIA-Season', 'PCIAT-Season', 'SDS-Season', 'PreInt_EduHx-Season'
    ]

    for col in season_cols:
        if col in test_data.columns:
            test_data[col] = test_data[col].map(season_mapping).fillna(-1).astype(int)

    test_data.fillna(0, inplace=True)
    X_test_data = test_data[common_columns].drop(columns=['id'], errors='ignore')
    return X_test_data
