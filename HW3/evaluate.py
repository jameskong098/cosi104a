import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_predictions(output_file, label_file):
    predicted_data = pd.read_csv(output_file)
    actual_data = pd.read_csv(label_file)
    
    if len(predicted_data) != len(actual_data):
        raise ValueError("Mismatch in number of rows between the predicted output and actual labels.")
    
    predicted_labels = predicted_data['Prediction']
    actual_labels = actual_data['Label']
    
    accuracy = accuracy_score(actual_labels, predicted_labels)
    print("\nComparing HW3_Test_Output.csv and HW3_Test_Label.csv...\n")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(actual_labels, predicted_labels))
    
    return accuracy

if __name__ == '__main__':
    output_file = 'HW3_Test_Output.csv'
    label_file = 'HW3_Test_Label.csv'
    
    evaluate_predictions(output_file, label_file)
