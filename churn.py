import pandas as pd

CSV_FILE = "Churn_Modelling.csv"
TARGET = "Exited"

data_frame = pd.read_csv(CSV_FILE)
data_frame

x = data_frame.drop(columns=TARGET)
y = data_frame[TARGET]

print(type(x))
print(type(y))
