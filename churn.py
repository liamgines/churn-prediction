import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

CSV_FILE = "Churn_Modelling.csv"
TARGET = "Exited"
UNNECESSARY_FEATURES = ["RowNumber", "CustomerId", "Surname"]
NOMINAL_FEATURES = ["Surname", "Geography", "Gender"]
RANDOM_STATE = 42

data_frame = pd.read_csv(CSV_FILE)

label_encoder = LabelEncoder()
for nominal_feature in NOMINAL_FEATURES:
    if nominal_feature in UNNECESSARY_FEATURES:
        pass

    else:
        current_feature_values = data_frame[nominal_feature]
        label_encoder.fit(current_feature_values)
        data_frame[nominal_feature] = label_encoder.transform(current_feature_values)

data_frame = data_frame.drop(columns=UNNECESSARY_FEATURES)

data_frame.plot(kind="box")
plt.xticks(rotation=90)
plt.show()

x = data_frame.drop(columns=[TARGET])
y = data_frame[TARGET]

def test_model(model):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=RANDOM_STATE, stratify=y)
    model.fit(x_train, y_train)
    y_test_predicted = model.predict(x_test)

    print(classification_report(y_test, y_test_predicted))
    return model
    
test_model(DecisionTreeClassifier(random_state=RANDOM_STATE))
