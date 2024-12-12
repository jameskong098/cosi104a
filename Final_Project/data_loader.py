import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def loadParquetFile(directory, fileName):
    """
    Read parquet file.
    """
    path = os.path.join(directory, fileName, "part-0.parquet")
    df = pd.read_parquet(path)
    df.drop("step", axis=1, inplace=True)  # drop step column
    statDF = df.describe().values.reshape(-1)
    return statDF, fileName.split("=")[1]  # get ids

def loadTimeSeriesData(directory):
    """
    Load time series data from parquet files.
    """
    filesIds = os.listdir(directory)  # get list of folder names (file ids)
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: loadParquetFile(directory, fname), filesIds), total=len(filesIds)))
    statistic, ids = zip(*results)  # pack into Statistic and Ids
    data = pd.DataFrame(statistic, columns=[f"stat_{i}" for i in range(len(statistic[0]))])
    data["id"] = ids  # add ids into dataframe
    return data

def load_data(train_file, test_file, train_parquet_dir, test_parquet_dir):
    """
    Load the training and test data from CSV and parquet files.

    Parameters:
    - train_file (str): Path to the training data CSV file.
    - test_file (str): Path to the test data CSV file.
    - train_parquet_dir (str): Path to the directory containing training parquet files.
    - test_parquet_dir (str): Path to the directory containing test parquet files.

    Returns:
    - X_train (DataFrame): Features of the training data.
    - y_train (Series): Target variable of the training data.
    - X_test (DataFrame): Features of the test data.
    - test_ids (Series): IDs of the test data.
    """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Drop rows with missing target values
    train_data = train_data.dropna(subset=['sii'])

    # Extract features and target variable
    X_train = train_data.drop(columns=['id', 'sii'])
    y_train = train_data['sii']
    X_test = test_data.drop(columns=['id'])
    test_ids = test_data['id']

    # Load parquet files and merge with CSV data
    train_parquet_data = loadTimeSeriesData(train_parquet_dir)
    test_parquet_data = loadTimeSeriesData(test_parquet_dir)

    combined_train_df = pd.merge(train_data, train_parquet_data, how="left", on='id')
    combined_test_df = pd.merge(test_data, test_parquet_data, how="left", on='id')

    X_train = combined_train_df.drop(columns=['id', 'sii'])
    y_train = combined_train_df['sii']
    X_test = combined_test_df.drop(columns=['id'])
    test_ids = combined_test_df['id']

    return X_train, y_train, X_test, test_ids