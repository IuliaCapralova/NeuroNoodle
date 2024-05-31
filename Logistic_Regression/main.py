from logistic_regression import LogisticRegression_1
from sklearn.linear_model import LogisticRegression
from standard_scaler import StandardScaler
import numpy as np
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

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # define parameter values (number of iter, learning rate)
    n_iter = 100
    alpha = 0.02

    # results of our model
    model = LogisticRegression_1(n_iter, alpha)
    model.train(X, y)
    y_pred = model.predict(X)
    accuracy = model.accuracy(y_pred, y)
    print("Our model:", accuracy)

    # sklearn's results
    sklearn_model = LogisticRegression(max_iter=n_iter, solver='lbfgs')
    sklearn_model.fit(X, y)
    sklearn_y_pred = sklearn_model.predict(X)
    sklearn_accuracy = np.mean(sklearn_y_pred == y)
    print(f"Sklearn accuracy: {sklearn_accuracy}")


if __name__ == "__main__":
    main()
