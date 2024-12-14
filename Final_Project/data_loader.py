import pandas as pd

def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
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
