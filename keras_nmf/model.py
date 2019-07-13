
import numpy as np
from keras import layers, regularizers, constraints, initializers, backend, models, optimizers
import tensorflow as tf


class PredictedIJ(layers.Layer):

    def __init__(self, k, **kwargs):
        self.k = k
        super(PredictedIJ, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lammbda = self.add_weight(
            name='lammbda',
            shape=(self.k, ),
            initializer=initializers.Constant(1 / self.k),
            constraint=constraints.NonNeg(),
            trainable=True
        )
        super(PredictedIJ, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # x_i is n x k, x_j is n x n_pairs x k
        x_i, x_j = x
        lammbda = tf.linalg.tensor_diag(self.lammbda)
        x_i = x_i @ lammbda
        return backend.squeeze(tf.matmul(x_i[:, None, :], x_j, transpose_b=True), axis=1)[:, :, None]

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], 1)


class NMFModel(object):

    def __init__(self, n: int, n_pairs: int, n_factors: int=None):
        assert n_factors is not None

        self.n = n
        self.n_pairs = n_pairs
        self.k = n_factors

    def compile_model(self, learning_rate):

        i_input = layers.Input(shape=(1, ), dtype='int32')
        ij_input = layers.Input(shape=(self.n_pairs, ), dtype='int32')
        self.W = layers.Embedding(
            self.n,
            self.k,
            embeddings_constraint=constraints.NonNeg(),
            embeddings_initializer=initializers.RandomUniform(minval=0, maxval=1),
            embeddings_regularizer=regularizers.l1(1e-3)
        )

        squeeze_layer = layers.Lambda(lambda x: backend.squeeze(x, axis=1))
        w_i = squeeze_layer(self.W(i_input))
        w_j = self.W(ij_input)

        predicted_ij = PredictedIJ(self.k, name='predicted_ij')([w_i, w_j])

        self.keras_model = models.Model(
            inputs=[i_input, ij_input],
            outputs=predicted_ij,
        )
        self.keras_model.compile(
            optimizers.Adam(lr=learning_rate),
            loss='mse',
            sample_weight_mode='temporal'
        )

    def fit(self, x_i, x_ij, y, masking_weights, epochs):
        return self.keras_model.fit([x_i, x_ij], y, sample_weight=masking_weights, epochs=epochs)
