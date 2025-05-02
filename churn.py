import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "Churn_Modelling.csv"
TARGET = "Exited"
UNNECESSARY_FEATURES = ["RowNumber", "CustomerId", "Surname"]

data_frame = pd.read_csv(CSV_FILE)
print(data_frame.isnull().sum())

x = data_frame.drop(columns=[TARGET])
y = data_frame[TARGET]

data_frame.plot(kind="box")
plt.xticks(rotation=90)
plt.show()
