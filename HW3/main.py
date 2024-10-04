from data_loader import load_data
from model import train_svm
import pandas as pd

def main():
    train_file = "HW3_Train.csv"
    test_file = "HW3_Test.csv"
    output_file = "HW3_Test_Output.csv"
    
    X_train, y_train, X_test = load_data(train_file, test_file)
    
    svm_model = train_svm(X_train, y_train)

    predictions = svm_model.predict(X_test)
    
    pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
