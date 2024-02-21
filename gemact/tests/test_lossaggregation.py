import unittest
import gemact

class TestLossModel(unittest.TestCase):
    def test_lossaggregation(self):
        decimalPlace = 3

        lossaggregate = gemact.LossAggregation(
            margins=gemact.Margins(
                dist=['genpareto', 'lognormal'],
                par=[{'loc': 0, 'scale': 1/.9, 'c': 1/.9}, {'loc': 0, 'scale': 10, 'shape': 1.5}],
                ),
                copula=gemact.Copula(
                    dist='frank',
                    par={'par': 1.2, 'dim': 2}
                    ),
                    n_sim=500000,
                    random_state=10,
                    n_iter=8
                    )

        self.assertAlmostEqual(lossaggregate.cdf(x=1, method='aep'), 0.021527811027268903, decimalPlace)

