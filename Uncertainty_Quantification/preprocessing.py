import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def load_data():
    # load and preprocess data
    base_path = os.getcwd() 
    file_name = "data/full_data.csv"
    full_path = os.path.join(base_path, file_name)
    data = pd.read_csv(full_path)
    df = data.drop(['id', 'Unnamed: 32'],axis = 1 )
    df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

    X = df.drop('diagnosis', axis = 1)
    y = df.diagnosis
    return X, y

def split_data(X, y, test_size, val_size):
    min_values = X.min()
    max_values = X.max()
    X = X.to_numpy()
    y = tf.keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)
    X_val = []
    y_val = []

    # scale the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    #X_val = scaler.transform(X_val) 
    X_test = scaler.transform(X_test)

    # augment data
    X_train, y_train = augment(X_train, y_train, noise_factor=0.01, N=2)
    #X_val, y_val = augment(X_val, y_val, noise_factor=0.01, N=0)

    # reduce dimensionality via PCA
    pca = PCA(n_components=13)
    X_train = pca.fit_transform(X_train)
    #X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    # generate random data as OOD (out of distribution) for evaluating uncertainty measures
    ood_data = np.zeros((num_ood, len(min_values)))
    for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
        ood_data[:, i] = np.random.uniform(low=min_val, high=max_val, size=X_test.shape[0])
    ood_data = scaler.transform(ood_data)
    ood_data = pca.transform(ood_data)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, ood_data

def augment(X, y, noise_factor, N):
    X_aug = [X]
    y_aug = [y]
    for i in range(N):
        X_aug.append(X + noise_factor * tf.random.normal(shape=X.shape))
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)