from logistic_regression import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # reduce data
    pca_sklearn = PCA(n_components=14)
    X_pca = pca_sklearn.fit_transform(X)

    # define parameter values (number of iter, learning rate)
    n_iter = 100
    alpha = 0.02

    # results of our model
    model = LogisticRegression(n_iter, alpha)
    model.train(X_pca, y)
    y_pred = model.predict(X_pca)
    accuracy = model.accuracy(y_pred, y)
    print("Model accuracy:", accuracy)

    # confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['B', 'M'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


if __name__ == "__main__":
    main()
