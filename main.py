from sklearn.model_selection import train_test_split
import pandas as pd

DATA_FILE_NAME = "Churn_Modelling.csv"
TARGET_COLUMN_NAME = "Exited"

def main():
    data = pd.read_csv(DATA_FILE_NAME)

    x = data.drop(columns=[TARGET_COLUMN_NAME])
    y = data[TARGET_COLUMN_NAME]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, train_size=0.2)

main()
