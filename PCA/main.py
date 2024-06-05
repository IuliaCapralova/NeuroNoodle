import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pca_class import PCA_1
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import pandas as pd


def main():

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

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # OUR PCA
    # firstly we want to see value of all variables, thus num_components is None
    # based on the calculated eigenvectors and values we want to sort them
    # and use those that expalain more than 0.98 precent of the variation :))
    pca_custom = PCA_1(plot=True)
    X_pca_custom = pca_custom.train_project(X)
    print("shape: ", X_pca_custom.shape)  # should be (569, 30)
    print("Transformed data from our PCA (not reduced yet):\n", X_pca_custom)

    # find the cumulative variance explained by the principal components
    # this thingy changes values of explained_variance so the sum up to 1
    cumulative_variance = np.cumsum(pca_custom.explained_variance)

    # here we find the index of the first occurrence of True in the boolean array
    # add one cuz it starts with index 0
    num_components = np.argmax(cumulative_variance >= 0.98) + 1
    print("num_components: ", num_components)

    pca_custom = PCA_1(k=num_components, plot=False)
    X_pca_custom = pca_custom.train_project(X)
    print("shape: ", X_pca_custom.shape)    # should be (569, 14)
    print("Reduced dataset:\n", X_pca_custom)

    # code below just to check if our PCA works fine
    print("---------------------------------")

    # scikit-learn PCA implementation

    # firslty with all 30 features
    pca_sklearn = PCA()
    X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)
    print("scikit-learn PCA data transromation (not reduced):\n", X_pca_sklearn)

    # find most important features
    explained_variance_ratio = pca_sklearn.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_variance_ratio >= 0.98) + 1
    print(f"Number of components to retain 98% variance: {num_components}")

    # PCA with 14 components
    pca_sklearn = PCA(n_components=num_components)
    X_pca = pca_sklearn.fit_transform(X_scaled)
    print("Transformed shape:", X_pca.shape)
    print("scikit-learn PCA Transformed Data reduced:\n", X_pca)


if __name__ == "__main__":
    main()
