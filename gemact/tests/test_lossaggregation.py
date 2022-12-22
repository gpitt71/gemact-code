# import unittest
# import gemact
#
# class TestDistributions(unittest.TestCase):
#     def test_aep(self):
        # la = gemact.LossAggregation(
        #     margins=['genpareto',
        #              'genpareto'],
        #     margins_pars=[
        #         {'loc': 0,
        #          'scale': 1 / .9,
        #          'c': 1 / .9},
        #         {'loc': 0,
        #          'scale': 1 / 1.8,
        #          'c': 1 / 1.8}
        #     ],
        #     copula='clayton',
        #     copula_par={'par': 1.2,
        #                 'dim': 2})
        #
        # decimalPlace=3
        #
        # self.assertAlmostEqual(la.cdf(1, method='aep'), , decimalPlace)


# if __name__ == '__main__':
#     unittest.main()