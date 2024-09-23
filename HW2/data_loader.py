import pandas as pd

def load_training_data(file_path):
    """
    Loads the training data from a CSV file.
    
    Args:
        file_path: str, path to the training data CSV.
    
    Returns:
        df: pd.DataFrame, loaded DataFrame.
    """
    df = pd.read_csv(file_path)

    # Clean "binnedInc" column as it is a string of ranged values
    df[['binnedInc1', 'binnedInc2']] = df['binnedInc'].str.extract(r'\((.*), (.*)\]')
    df['binnedInc1'] = df['binnedInc1'].astype(float)
    df['binnedInc2'] = df['binnedInc2'].astype(float)
    df = df.drop(columns=['binnedInc'])
    
    # Drop the 'Geography' column since it's categorical and not useful for regression
    df = df.drop(columns=['Geography'])

    target_columns = ['avgAnnCount', 'avgDeathsPerYear', 'TARGET_deathRate', 'incidenceRate']
    
    # Define the most relevant features for each target variable
    feature_sets = {
        'avgAnnCount': ['medIncome', 'popEst2015', 'povertyPercent', 'MedianAge', 'PercentMarried', 'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate', 'binnedInc1', 'binnedInc2'],
        'avgDeathsPerYear': ['popEst2015', 'povertyPercent', 'studyPerCap', 'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace'],
        'TARGET_deathRate': ['medIncome', 'popEst2015', 'povertyPercent', 'studyPerCap', 'MedianAge', 'MedianAgeMale', 'MedianAgeFemale', 'AvgHouseholdSize', 'PercentMarried', 'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate', 'binnedInc1', 'binnedInc2'],
        'incidenceRate': ['studyPerCap', 'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace']
    }

    df = df.dropna()  
    
    return df, target_columns, feature_sets

def load_test_data(file_path):
    """
    Loads the test data from a CSV file.
    
    Args:
        file_path: str, path to the test data CSV.
    
    Returns:
        df: pd.DataFrame, loaded DataFrame.
    """
    df = pd.read_csv(file_path)

    # Clean "binnedInc" column as it is a string of ranged values
    df[['binnedInc1', 'binnedInc2']] = df['binnedInc'].str.extract(r'\((.*), (.*)\]')
    df['binnedInc1'] = df['binnedInc1'].astype(float)
    df['binnedInc2'] = df['binnedInc2'].astype(float)
    df = df.drop(columns=['binnedInc'])
    
    # Drop the 'Geography' column since it's categorical and not useful for regression
    df = df.drop(columns=['Geography'])

    df = df.dropna()  
    
    return df
