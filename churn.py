import pandas as pd

CSV_FILE = "Churn_Modelling.csv"
TARGET = "Exited"
UNNECESSARY_FEATURES = ["RowNumber", "CustomerId", "Surname"]

data_frame = pd.read_csv(CSV_FILE)

x = data_frame.drop(columns=[TARGET])
y = data_frame[TARGET]
