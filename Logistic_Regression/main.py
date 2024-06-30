from logistic_regression import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pca_class import PCA_1


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # scale dara
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # reduce data
    pca = PCA_1(k=13)
    X_train_pca = pca.train_project(X_train)
    X_test_pca = pca.train_project(X_test)

    # define parameter values (number of iter, learning rate)
    n_iter = 100
    alpha = 0.02

    # results of our model
    model = LogisticRegression(n_iter, alpha)
    model.train(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    accuracy = model.accuracy(y_pred, y_test)
    print("Model accuracy:", accuracy)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['B', 'M'])
    disp.plot(cmap=plt.cm.Blues)

    plt.rcParams.update({'font.size': 12})
    disp.ax_.set_xlabel('Predicted label', fontsize=12.1)
    disp.ax_.set_ylabel('True label', fontsize=12.1)
    disp.ax_.set_title('Confusion Matrix', fontsize=15)
    for labels in disp.text_.ravel():
        labels.set_fontsize(11)

    plt.show()


if __name__ == "__main__":
    main()
