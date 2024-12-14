import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(train_data, test_data):
    threshold = 0.5 * len(train_data)
    columns_with_data = train_data.columns[train_data.isnull().sum() < threshold]
    train_data = train_data[columns_with_data]
    train_data.fillna(0, inplace=True)
    train_data.dropna(subset=['sii'], inplace=True)

    season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    season_cols = [
        'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season',
        'FGC-Season', 'BIA-Season', 'PCIAT-Season', 'SDS-Season', 'PreInt_EduHx-Season'
    ]

    for col in season_cols:
        if col in train_data.columns:
            train_data[col] = train_data[col].map(season_mapping).fillna(-1).astype(int)

    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    correlation_with_label = train_data.corr()['sii']
    corr_threshold = 0.2
    high_corr_columns = correlation_with_label[abs(correlation_with_label) >= corr_threshold].index
    train_data = train_data[high_corr_columns]

    common_columns = train_data.columns.intersection(test_data.columns)
    X = train_data[common_columns]
    y = train_data['sii']

    encoder = LabelEncoder()
    categorical_columns = X.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        X[col] = encoder.fit_transform(X[col].astype(str))

    X.drop(columns=['id'], errors='ignore', inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, common_columns, scaler, pca
