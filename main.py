from sklearn.model_selection import train_test_split
import pandas as pd

DATA_FILE_NAME = "Churn_Modelling.csv"
TARGET_COLUMN_NAME = "Exited"

def main():
    # https://stackoverflow.com/questions/10867028/get-pandas-read-csv-to-read-empty-values-as-empty-string-instead-of-nan
    data = pd.read_csv(DATA_FILE_NAME, keep_default_na=False)

    x = data.drop(columns=[TARGET_COLUMN_NAME])
    y = data[TARGET_COLUMN_NAME]

    feature_names = data.columns.values # [feature_name for feature_name in data]
    for feature_name in feature_names:
        print("" in data[feature_name])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, train_size=0.2)

main()
