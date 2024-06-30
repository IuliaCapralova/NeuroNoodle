import numpy as np
import keras
from sklearn.model_selection import KFold
import itertools
#from PCA.pca_class import PCA_1

from stochastic_classifier import StochasticClassifier, basic_model
from calibration import calibration
from plotting import confmatrix, plot_entropies
from preprocessing import load_data, split_data


def main():
    keras.utils.set_random_seed(0)
    X, y = load_data()
    
    X_train, y_train, X_val, y_val, X_test, y_test, ood_data_pca = split_data(X, y, test_size=0.25, val_size=0.05)
    stoch = StochasticClassifier(
        X_train=X_train,
        y_train=y_train,
        #validations=(X_val, y_val),
        num_samples=100,
        params=(0.0005, 10, 10, 0.1)
        )
    stoch.plot_history()

    #cross_validation(X, y)

    # MAKE PREDICTIONS ON TEST SET & DISENTANGLE UNCERTAINTIES
    prob_ale, prob_epi, unc_ale, unc_epi = stoch.disentangle(X_test)

    # CONFUSION MATRIX
    y_pred_ale = np.argmax(prob_ale, axis=1)
    y_pred_epi = np.argmax(prob_epi, axis=1)
    y_true = np.argmax(y_test, axis=1)
    confmatrix(y_true, y_pred_ale, normalized=False)
    confmatrix(y_true, y_pred_epi, normalized=False)

    # CHECK CALIBRATION AND RELIABLITY PLOT
    #calibration(y_test, prob_ale, unc_ale, num_bins=10, conf_type="entropy")
    #calibration(y_test, prob_ale, unc_ale, num_bins=10, conf_type="maxprob")
    calibration(y_test, prob_epi, unc_epi, num_bins=10, conf_type="entropy")
    #calibration(y_test, prob_epi, unc_epi, num_bins=10, conf_type="maxprob")

    # POSSIBILITIES FOR OOD DETECTION
    _, _, _, unc_epi_ood = stoch.disentangle(ood_data_pca)
    plot_entropies(unc_epi, unc_epi_ood)

    # COMPARE TO BASIC MODEL
    basic = basic_model(X_train, y_train)
    basic_pred = basic(X_test)
    entropy = -np.sum(basic_pred * np.log2(basic_pred + 1e-10), axis=1)
    plot_entropies(unc_epi, entropy, labels=['Uncertainty model', 'Basic model'])

def cross_validation(X, y):

    X_train, y_train, X_val, y_val, X_test, y_test, ood_data_pca = split_data(X, y, test_size=0.2, val_size=0.2)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    learning_rates = [0.0005, 0.001, 0.01]
    units1 = [20, 10, 5]
    units2 = [15, 10, 5]
    dropout_rate = [0.1, 0.2, 0.3]
    params = list(itertools.product(learning_rates, units1, units2, dropout_rate))

    validation_loss = {p: [] for p in params}
    for p in params:
        print(f"Evaluating model for params: {p}")

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            model = StochasticClassifier(X_train_fold, y_train_fold, num_samples=100, validations=(X_val_fold, y_val_fold), params = p)
            model.plot_history()
            history = model.get_history()
            val_loss = history.history['val_loss'][-1]
            validation_loss[p].append(val_loss)

    avg_loss = {p: np.mean(accs) for p, accs in validation_loss.items()}
    best_p = min(avg_loss, key=avg_loss.get)
    print("validation loss per params:", validation_loss)
    print("avg loss per params:", avg_loss)
    print("best params:", best_p)

def tuning(params):
    X, y = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, ood_data_pca = split_data(X, y, test_size=0.2, val_size=0.2)

    validation_loss = {p: [] for p in params}
    for p in params:
        print(f"Evaluating model for params: {p}")
        model = StochasticClassifier(X_train, y_train, num_samples=100, validations=(X_val, y_val), params = p)
        model.plot_history()
        history = model.get_history()
        val_loss = history.history['val_loss'][-1]
        validation_loss[p].append(val_loss)

    best_p = min(validation_loss, key=validation_loss.get)
    print("validation loss per params:", validation_loss)
    print("best params:", best_p)


if __name__ == "__main__":
    main()