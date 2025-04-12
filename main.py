from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DATA_FILE_NAME = "Churn_Modelling.csv"
TARGET_COLUMN_NAME = "Exited"
IRRELEVANT_COLUMN_NAMES = ["RowNumber", "CustomerId", "Surname"]
NON_NUMERIC_COLUMN_NAMES = ["Gender", "Geography"]

def digitize_column(data_frame, column_name):
    value_map = {}
    assigned_number = 0
    for value in data_frame[column_name]:
        if value not in value_map:
            value_map[value] = assigned_number
            assigned_number += 1

    data_frame[column_name] = data_frame[column_name].map(value_map)

def box_plot_frame(data_frame):
    data_frame.plot(kind="box")
    plt.xticks(rotation=60)
    plt.show()

def correlation_plot_frame(data_frame, feature_names):
    correlation_matrix = data_frame.corr()
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes_image = axes.matshow(correlation_matrix, interpolation="nearest")
    figure.colorbar(axes_image)
    axes.set_xticks([i for i in range(len(feature_names))])
    axes.set_yticks([i for i in range(len(feature_names))])
    axes.set_xticklabels(feature_names)
    axes.set_yticklabels(feature_names)
    plt.xticks(rotation=90)
    plt.show()

def main():
    data_frame = pd.read_csv(DATA_FILE_NAME)
    data_frame = data_frame.drop(columns=IRRELEVANT_COLUMN_NAMES)
    
    feature_names = data_frame.columns.values # [feature_name for feature_name in data_frame]

    for column_name in NON_NUMERIC_COLUMN_NAMES:
        digitize_column(data_frame, column_name)

    print(data_frame)

    box_plot_frame(data_frame)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_frame)

    data_frame = pd.DataFrame(data=scaled_data, columns=feature_names)

    box_plot_frame(data_frame)
    
    correlation_plot_frame(data_frame, feature_names)

    x = data_frame.drop(columns=[TARGET_COLUMN_NAME])
    y = data_frame[TARGET_COLUMN_NAME]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, train_size=0.2)

main()
