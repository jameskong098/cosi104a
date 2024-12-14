import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

def loadParquetFile(directory, fileName):
    """
    Read parquet file.
    """
    path = os.path.join(directory, fileName, "part-0.parquet")
    df = pd.read_parquet(path)
    df.drop("step", axis=1, inplace=True)  
    statDF = df.describe().values.reshape(-1)
    return statDF, fileName.split("=")[1]  

def loadTimeSeriesData(directory):
    """
    Load time series data from parquet files.
    """
    filesIds = os.listdir(directory) 
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: loadParquetFile(directory, fname), filesIds), total=len(filesIds)))
    statistic, ids = zip(*results)  
    data = pd.DataFrame(statistic, columns=[f"stat_{i}" for i in range(len(statistic[0]))])
    data["id"] = ids  
    return data

def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    
    train_data = train_data.dropna(subset=['sii'])

    train_parquet_data = loadTimeSeriesData("series_train.parquet")
    test_parquet_data = loadTimeSeriesData('series_test.parquet')

    train_data = pd.merge(train_data, train_parquet_data, how="left", on='id')
    test_data = pd.merge(test_data, test_parquet_data, how="left", on='id')

    return train_data, test_data, sample_submission

def prepare_test_data(test_data, common_columns):
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
