from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

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

def box_plot_frame(data_frame, title=""):
    data_frame.plot(kind="box")
    plt.title(title)
    plt.xticks(rotation=60)
    plt.show()

def correlation_plot_frame(data_frame):
    correlation_matrix = data_frame.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def analyze_components(data_frame):
    threshold = 0.95

    component_analysis = PCA()
    component_analysis.fit(data_frame)    
    
    def components_to_retain(explained_variance_ratio, threshold):
        cumulative_variance_sums = np.cumsum(explained_variance_ratio)
        
        component_count = len(cumulative_variance_sums)
        for i in range(len(cumulative_variance_sums)):     
            if threshold <= cumulative_variance_sums[i]:   
                component_count = i + 1                    
                break                                      
                
        return component_count   
    
    component_counts = [i for i in range(1, len(np.cumsum(component_analysis.explained_variance_ratio_)) + 1)]
    summed_explained_variance_ratios = np.cumsum(component_analysis.explained_variance_ratio_)
    plt.plot(component_counts, summed_explained_variance_ratios, "-o")
    plt.title("Principal Components Analysis")
    plt.xlabel("Components")
    plt.ylabel("Explained Variance Ratio")
    plt.show()
    
    component_count = components_to_retain(component_analysis.explained_variance_ratio_, threshold)   

    n_components_analysis = PCA(n_components=component_count)   
    n_components_analysis.fit(data_frame)   
    
    print("Explained Variance Ratio for each Principal Component:\n")
    for i in range(len(n_components_analysis.explained_variance_ratio_)):    
        print(f"Component {i + 1}: {n_components_analysis.explained_variance_ratio_[i] * 100:.2f}%")

    print("\nPCA captures {:.2f}% of the variance with {} components.".format(sum(n_components_analysis.explained_variance_ratio_) * 100, component_count)) 

def main():
    data_frame = pd.read_csv(DATA_FILE_NAME)
    data_frame = data_frame.drop(columns=IRRELEVANT_COLUMN_NAMES)
    
    feature_names = data_frame.columns.values # [feature_name for feature_name in data_frame]

    for column_name in NON_NUMERIC_COLUMN_NAMES:
        digitize_column(data_frame, column_name)

    box_plot_frame(data_frame, "Unscaled Data")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_frame)

    data_frame = pd.DataFrame(data=scaled_data, columns=feature_names)

    box_plot_frame(data_frame, "Scaled Data")
    
    correlation_plot_frame(data_frame)

    analyze_components(data_frame)

    x = data_frame.drop(columns=[TARGET_COLUMN_NAME])
    y = data_frame[TARGET_COLUMN_NAME]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, train_size=0.2)

main()
