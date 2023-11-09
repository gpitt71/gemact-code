import unittest
import numpy as np
from gemact import copulas

class Test_Discrete_Distributions(unittest.TestCase):

    def test_ClaytonCopula(self):
        copula = copulas.ClaytonCopula(par=1.2, dim=2)
        vals = np.array([[.5, .5]])

        self.assertAlmostEqual(copula.cdf(vals)[0], 0.3443011, 3)




