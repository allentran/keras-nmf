import unittest

import numpy as np
from keras_nmf import NMFModel


class TestFitRandom(unittest.TestCase):

    def test_decrease_loss(self):
        nmf_model = NMFModel(99, 7, 4)
        nmf_model.compile_model()

        ii = np.random.randint(0, 99, 128)
        jj = np.random.randint(0, 99, (128, 7))
        y = np.random.uniform(1, 6, (128, 7, 1))
        weights = np.random.uniform(1e-5, 1, (128, 7)).astype('float32')
        history = nmf_model.fit(ii, jj, y, weights, epochs=100).history

        self.assertLess(history['loss'][-1], history['loss'][0])
