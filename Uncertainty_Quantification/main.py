import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns

from stochastic_classifier import StochasticClassifier
from calibration import calibration

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
    y_tf = tf.keras.utils.to_categorical(y)

    # scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # reduce dimensionality via PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    # split the data
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y_tf, test_size=0.2, random_state=42)

    # generate augmented testing data for evaluating uncertainty measures
    noise_factor = 0.1
    X_test_noisy = X_test_pca + noise_factor * tf.random.normal(shape=X_test_pca.shape)
    X_test_noisy = tf.clip_by_value(X_test_noisy, 0.0, 1.0)
    X_test_combined = tf.concat([X_test_pca, X_test_noisy], axis=0)
    y_test_combined = tf.concat([y_test, y_test], axis=0)

    # generate random data as OOD (out of distribution) for evaluating uncertainty measures
    np.random.seed(42)
    ood_data = np.random.rand(100, 30)  # 100 samples, 30 features
    ood_data_scaled = scaler.transform(ood_data)
    ood_data_pca = pca.transform(ood_data_scaled)

    return X_train_pca, X_test_pca, X_test_combined, y_train, y_test, y_test_combined, ood_data_pca

def main():
    X_train_pca, X_test_pca, X_test_combined, y_train, y_test, y_test_combined, ood_data_pca = load_data()
    stoch = StochasticClassifier(
        X_train=X_train_pca,
        y_train=y_train,
        validations=None,
        num_samples=100,
        lr=0.0005)
    stoch.plot_history()

    #cross_validation(X_train_pca, y_train)

    # MAKE PREDICTIONS ON TEST SET & DISENTANGLE UNCERTAINTIES
    #prob_ale, prob_epi, unc_ale, unc_epi = stoch.disentangle(X_test_pca)
    prob_ale, prob_epi, unc_ale, unc_epi = stoch.disentangle(X_test_combined)

    # CONFUSION MATRIX
    y_pred_ale = np.argmax(prob_ale, axis=1)
    y_pred_epi = np.argmax(prob_epi, axis=1)
    y_true = np.argmax(y_test_combined, axis=1)
    confmatrix(y_true, y_pred_ale, normalized=False)
    confmatrix(y_true, y_pred_epi, normalized=False)

    # CHECK CALIBRATION AND RELIABLITY PLOT
    #calibration(y_test_combined, prob_ale, unc_ale, num_bins=10, conf_type="entropy")
    calibration(y_test_combined, prob_epi, unc_epi, num_bins=10, conf_type="entropy")

    # POSSIBILITIES FOR OOD DETECTION
    _, _, _, unc_epi_ood = stoch.disentangle(ood_data_pca)
    plot_ood(unc_epi, unc_epi_ood)

def cross_validation(X_train, y_train):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    learning_rates = [0.0001, 0.1]
    validation_accuracies = {lr: [] for lr in learning_rates}
    for lr in learning_rates:
        print(f"Evaluating model with learning rate: {lr}")
        fold_no = 1

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            model = StochasticClassifier(X_train_fold, y_train_fold, num_samples=100, lr=lr, validations=(X_val_fold, y_val_fold))
            history = model.get_history()
            val_accuracy = history.history['val_accuracy'][-1]
            validation_accuracies[lr].append(val_accuracy)
            fold_no += 1

    avg_accuracies = {lr: np.mean(accs) for lr, accs in validation_accuracies.items()}
    best_lr = max(avg_accuracies, key=avg_accuracies.get)
    print("validation accuracies per lr:", validation_accuracies)
    print("avg acc per lr:", avg_accuracies)
    print("best lr:", best_lr)

def confmatrix(y_true, y_pred, normalized=True):        
    cm = confusion_matrix(y_true, y_pred)

    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmtt = '.2%'
    else:
        fmtt = 'd'

    annot = np.empty_like(cm).astype(str)
    n_classes = cm.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            if i == 0 and j == 0:
                annot[i, j] = f'TN\n{cm[i, j]:{fmtt}}'
            elif i == 0 and j == 1:
                annot[i, j] = f'FP\n{cm[i, j]:{fmtt}}'
            elif i == 1 and j == 0:
                annot[i, j] = f'FN\n{cm[i, j]:{fmtt}}'
            elif i == 1 and j == 1:
                annot[i, j] = f'TP\n{cm[i, j]:{fmtt}}'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_ood(id_epi, ood_epi):
    bins = np.arange(0, 1.1, 0.1)

    plt.hist(id_epi, bins=bins, color='red', alpha=0.5, label='ID')
    plt.hist(ood_epi, bins=bins, color='blue', alpha=0.5, label='OOD')

    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Entropy in ID vs OOD')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()