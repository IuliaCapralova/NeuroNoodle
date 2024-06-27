import numpy as np
import pandas as pd
from pca_class import PCA_1
from sklearn.preprocessing import MinMaxScaler
import os
import warnings


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    base_path = os.getcwd() 
    file_name = "data/full_data.csv"
    full_path = os.path.join(base_path, file_name)

    data = pd.read_csv(full_path)

    # clean data
    df = data.drop(['id', 'Unnamed: 32'],axis = 1 )
    df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

    X = df.drop('diagnosis', axis = 1)
    y = df.diagnosis

    feature_names = X.columns.tolist()

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # OUR PCA
    # firstly we want to see value of all variables
    # based on the calculated eigenvectors and values we want to sort them
    # and use those that expalain more than 0.98 precent of the variation
    pca_custom = PCA_1(plot=True, feature_names=feature_names)
    X_pca_custom = pca_custom.train_project(X_scaled)
    print("shape: ", X_pca_custom.shape)  # should be (569, 30)

    # find the cumulative variance explained by the principal components
    # this changes values of explained_variance so the sum up to 1
    cumulative_variance = np.cumsum(pca_custom.explained_variance)

    # here we find the index of the first occurrence of True in the boolean array
    # add one since it starts with index 0
    num_components = np.argmax(cumulative_variance >= 0.98) + 1
    print("num_components: ", num_components)

    pca_custom = PCA_1(k=num_components, plot=False)
    X_pca_custom = pca_custom.train_project(X)
    print("shape: ", X_pca_custom.shape)
    print("Reduced dataset:\n", X_pca_custom)


if __name__ == "__main__":
    main()
