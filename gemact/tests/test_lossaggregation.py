import unittest
import gemact

class TestLossModel(unittest.TestCase):
    def test_lossaggregation(self):
        decimalPlace = 3

        lossaggregate = gemact.LossAggregation(
            margins=['genpareto', 'genpareto'],
            margins_pars=[
                {'loc': 0,
                 'scale': 1 / .9,
                 'c': 1 / .9},
                {'loc': 0,
                 'scale': 1 / 1.8,
                 'c': 1 / 1.8}
            ],
            copula='clayton',
            copula_par={'par': 1.2, 'dim': 2}
        )

        self.assertAlmostEqual(lossaggregate.cdf(1, n_iter=7, method='aep'), 0.31583504136297336, decimalPlace)

