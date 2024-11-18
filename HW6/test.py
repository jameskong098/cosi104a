import pandas as pd
from sklearn.metrics import f1_score

# Load the CSV file
file_path = 'test_label.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Define the function to compute F1 score
def compute_f1_score(df):
    """
    Compute the F1 score by comparing the class labels with themselves.
    
    Args:
        df (pd.DataFrame): DataFrame containing the class labels in the 'Class' column.
        
    Returns:
        float: F1 score (comparing the column with itself).
    """
    # Extract the true and predicted labels (both are the same in this case)
    true_labels = df['Class']
    predicted_labels = df['Class']
    
    # Compute the F1 score
    f1 = f1_score(true_labels, predicted_labels, average='macro')  # Macro-average for multi-class F1
    
    return f1

# Calculate the F1 score
f1_result = compute_f1_score(data)
print("F1 Score:", f1_result)
