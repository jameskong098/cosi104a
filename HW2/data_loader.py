import pandas as pd

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and creating dummy variables.
    
    Args:
        df: pd.DataFrame, the dataset to preprocess.
    
    Returns:
        df: pd.DataFrame, the preprocessed dataset.
    """
    # Fill numerical columns with their median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with their mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Create dummy variables for categorical columns (binnedInc, Geography)
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def load_training_data(file_path):
    df = pd.read_csv(file_path)
    df = preprocess_data(df) 

    target_columns = ['avgAnnCount', 'avgDeathsPerYear', 'TARGET_deathRate', 'incidenceRate']
    
    # Define the most relevant features for each target variable
    feature_sets = {
        'avgAnnCount': ['popEst2015', 'PercentMarried', 'PctNoHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctPrivateCoverage', 'PctPublicCoverage', 'PctMarriedHouseholds'],
        'avgDeathsPerYear': ['popEst2015', 'povertyPercent', 'studyPerCap', 'PctNoHS18_24', 'PctHS18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctWhite', 'PctBlack'],
        'TARGET_deathRate': ['medIncome', 'popEst2015', 'povertyPercent', 'studyPerCap', 'MedianAgeFemale', 'PercentMarried', 'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate'],
        'incidenceRate': ['studyPerCap', 'PctNoHS18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace']
    }

    return df, target_columns, feature_sets

def load_test_data(file_path):
    df = pd.read_csv(file_path)
    df = preprocess_data(df)  

    return df
