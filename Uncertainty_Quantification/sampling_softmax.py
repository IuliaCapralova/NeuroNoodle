import tensorflow as tf
import keras
from keras.layers import Layer

# NOTE this class is inspired and builds upon the code provided in:
# Tutorial for Uncertainty Disentanglement in Classification
# (RUG, Uncertainty in Machine Learning, 2023/24)

class SamplingSoftmax(Layer):
    '''
    Implements a softmax layer with sampling.
    A Gaussian distribution over the logits is defined from the input parameters
    (logit mean and logit variance), from which num_softmax_samples logits are
    sampled. Softmax is applied to each, and averaged into a probability output.
    '''
    def __init__(self, num_softmax_samples=200):
        self.num_softmax_samples = num_softmax_samples
        super().__init__()

    def call(self, inputs):
        assert len(inputs) == 2
        logit_mean = inputs[0]
        logit_var = inputs[1]

        # (batch_size, num_classes) to (batch_size, 1, num_classes)
        logit_mean = tf.expand_dims(logit_mean, axis=1)
        # create multiple (same) samples of the means across this dimension
        logit_mean = tf.keras.backend.repeat_elements(logit_mean, self.num_softmax_samples, axis=1)

        logit_var = tf.expand_dims(logit_var, axis=1)
        logit_var = tf.keras.backend.repeat_elements(logit_var, self.num_softmax_samples, axis=1)

        samples = self.sampling(logit_mean, logit_var)

        # apply softmax to each sample and return the sample mean as the probability P(y|x)
        softmaxed = tf.keras.backend.softmax(samples)
        prob = tf.keras.backend.mean(softmaxed, axis=1)

        return prob
    
    def sampling(self, logit_mean, logit_var):
        '''
        Samples logits from a Gaussian distribution defined
        by logit_mean and logit_var.
        '''
        normal_samples = tf.keras.backend.random_normal(tf.keras.backend.shape(logit_mean))
        samples = normal_samples * logit_var + logit_mean
        return samples