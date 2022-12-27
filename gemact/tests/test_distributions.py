import unittest
import gemact.distributions as distributions

class TestDistributions(unittest.TestCase):

    def test_Poisson(self):
        dist = distributions.Poisson(mu=10)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.03783327, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.8), 13.0, decimalPlace)
        self.assertAlmostEqual(dist.cdf(13), 0.8644644, decimalPlace)

    def test_ZTPoisson(self):
        dist = distributions.ZTPoisson(mu=10)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.03783499, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.95), 15.0, decimalPlace)
        self.assertAlmostEqual(dist.cdf(13), 0.8644583, decimalPlace)

    # def test_Binom(self):



if __name__ == '__main__':
    unittest.main()