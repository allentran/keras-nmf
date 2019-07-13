import unittest

import numpy as np
from keras_nmf import NMFModel


class TestFitRandom(unittest.TestCase):

    def test_decrease_loss(self):
        nmf_model = NMFModel(99, 7, 4)
        nmf_model.compile_model(learning_rate=0.5)

        ii = np.random.randint(0, 99, 128)
        jj = np.random.randint(0, 99, (128, 7))
        y = np.random.uniform(1, 6, (128, 7, 1))
        weights = np.random.uniform(1e-5, 1, (128, 7)).astype('float32')
        history = nmf_model.fit(ii, jj, y, weights, epochs=100).history

        self.assertLess(history['loss'][-1], history['loss'][0])

    def test_overfit(self):
        np.random.seed(1692)
        n = 16
        k = 8
        n_pairs = 9
        nmf_model = NMFModel(n, n_pairs, k)
        nmf_model.compile_model(learning_rate=0.25)

        latent = np.random.uniform(0, 4, size=(n, int(k / 2))) * np.random.randint(0, 2, (n, int(k / 2)))
        X = latent @ latent.T

        ii = np.arange(0, n).astype('int32')
        jj = np.repeat(np.random.choice(n, n_pairs, replace=False)[None, :], n, 0)
        y = X[ii[:, None], jj, None]
        history = nmf_model.fit(ii, jj, y, epochs=1000, masking_weights=None).history

        self.assertLess(history['loss'][-1], 0.05)

