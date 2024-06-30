import tensorflow as tf
import numpy as np
import keras
from keras import Sequential, Model
from keras.layers import Input, Dense, Dropout, Layer
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt

from sampling_softmax import SamplingSoftmax

# NOTE this code is inspired and builds upon the code provided in:
# Tutorial for Uncertainty Disentanglement in Classification
# (RUG, Uncertainty in Machine Learning, 2023/24)

class StochasticClassifier():
    '''
    Applies MC Dropout and uncertainty disentanglement to obtain predictions
    with corresponding uncertainties (aleatoric and epistemic).
    '''
    def __init__(self, X_train, y_train, validations=None, num_samples=10, params = (0.0005, 10, 10, 0.1)):
        '''
        Args:
        X_train, y_train = training data
        validations = validation datasets for CV
        num_samples = number of model samples in MC Dropout
        lr = learning rate
        '''
        self._history = None
        self._params = params
        self._model = self._train_model(X_train=X_train, y_train=y_train, validations=validations)
        self._num_samples = num_samples

    
    def _train_model(self, X_train, y_train, validations):
        '''
        Creates and trains a classifier with dropout layers.
        Implements softmax sampling.
        Returns a model that produces logit mean and var.
        '''
        input = Input(shape=(X_train.shape[1],))
        # hidden layers:
        x = Dense(units=self._params[1], activation='relu')(input)
        x = Dropout(rate=self._params[3])(x)
        x = Dense(units=self._params[2], activation='relu')(x)
        x = Dropout(rate=self._params[3])(x)
        # outputs:
        logit_mean = Dense(units=2, activation='linear')(x)
        logit_var = Dense(units=2, activation='softplus')(x)
        softmax_output = SamplingSoftmax()([logit_mean, logit_var])
        
        # train the model with the probability output
        model = Model(input, softmax_output)
        model.compile(optimizer=Adam(learning_rate=self._params[0]), 
                        loss=BinaryCrossentropy(),
                        metrics=['accuracy'])
        self._history = model.fit(X_train, y_train, validation_data=validations, epochs=300, batch_size=32)

        # return a model that produces the logit mean and var
        unc_model = Model(input, [logit_mean, logit_var])
        return unc_model

    def _MC_dropout(self, X_test):
        '''
        MC Dropout procedure.
        Get M predictions of mean and variance with dropout applied during inference.
        '''
        means = []
        vars = []
        for i in range(self._num_samples):
            logit_mean_pred, logit_var_pred = self._model(X_test, training=True)
            means.append(logit_mean_pred)
            vars.append(logit_var_pred)
        # logit means and vars across M models:
        stacked_means = tf.stack(means)
        stacked_vars = tf.stack(vars)
        return stacked_means, stacked_vars

    def disentangle(self, X_test):
        '''
        Disentangles aleatoric and epistemic uncertainties from mean and var
        predictions obtained from the MC Dropout procedure.
        '''

        stacked_means, stacked_vars = self._MC_dropout(X_test)
        # mean prediction logit across M models:
        pred_mean = tf.reduce_mean(stacked_means, axis=0)

        # aleatoric uncertainty logit = mean of the variances
        var_ale = tf.reduce_mean(stacked_vars, axis=0)
        prob_ale = SamplingSoftmax()([pred_mean, var_ale])
        # shannon entropy as the uncertainty/confidence measure
        unc_ale = - prob_ale * (tf.math.log(prob_ale + 1e-10) / tf.math.log(tf.constant(2.0, dtype=prob_ale.dtype)))
        unc_ale = [sum(ent) for ent in unc_ale.numpy()]

        # epistemic uncertainty logit = variance of the means
        var_epi = tf.math.reduce_variance(stacked_means, axis=0)
        prob_epi = SamplingSoftmax()([pred_mean, var_epi])
        # shannon entropy as the uncertainty/confidence measure
        unc_epi = - prob_epi * (tf.math.log(prob_epi + 1e-10) / tf.math.log(tf.constant(2.0, dtype=prob_epi.dtype)))
        unc_epi = [sum(ent) for ent in unc_epi.numpy()]

        # probability predictions and corresponding uncertainties
        return prob_ale, prob_epi, unc_ale, unc_epi
    
    def get_history(self):
        return self._history

    def plot_history(self):
        # plot training accuracy
        plt.plot(self._history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()

        # plot training and validation loss
        plt.plot(self._history.history['loss'], label='training loss')
        if 'val_loss' in self._history.history:
            plt.plot(self._history.history['val_loss'], label='validation loss')
        plt.title('training and validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

def basic_model(X_train, y_train):
    input = Input(shape=(X_train.shape[1],))
    x = Dense(units=10, activation='relu')(input)
    x = Dropout(rate=0.1)(x)
    x = Dense(units=10, activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    output = Dense(units=2, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.0005), 
                    loss=BinaryCrossentropy(),
                    metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=300, batch_size=32)
    return model