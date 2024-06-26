from logistic_regression import LogisticRegression
from standard_scaler import StandardScaler
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
    model = LogisticRegression(n_iter, alpha)
    model.train(X, y)
    y_pred = model.predict(X)
    accuracy = model.accuracy(y_pred, y)
    print("Model accuracy:", accuracy)


if __name__ == "__main__":
    main()
