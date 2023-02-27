import unittest
import matplotlib.pyplot as plt
import numpy as np
import gemact.distributions as distributions

class Test_Discrete_Distributions(unittest.TestCase):

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

    def test_ZMPoisson(self):
        dist = distributions.ZMPoisson(mu=5, p0m=.856)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.02543871, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.9), 4, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.5), 0.856, decimalPlace)

    def test_Binom(self):
        dist = distributions.Binom(n=10, p=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.2460938, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.5), 5, decimalPlace)
        self.assertAlmostEqual(dist.cdf(5), 0.6230469, decimalPlace)

    def test_ZTBinom(self):
        dist = distributions.ZTBinom(n=10, p=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.2463343, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.5), 5, decimalPlace)
        self.assertAlmostEqual(dist.cdf(5), 0.6226784, decimalPlace)

    # def test_ZMBinom(self):
    #     dist = distributions.ZMBinom(n=10, p=.5, p0m=.152)
    #     decimalPlace = 3
    #
    #     self.assertAlmostEqual(dist.pmf(5), 0.2155425, decimalPlace)
    #     self.assertAlmostEqual(dist.ppf(.5), 5, decimalPlace)
    #     self.assertAlmostEqual(dist.cdf(5), 0.6230469, decimalPlace)

    def test_Geom(self):
        dist = distributions.Geom(p=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.03125, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.5), 1., decimalPlace)
        self.assertAlmostEqual(dist.cdf(5), 0.96875, decimalPlace)

    def test_NegBinom(self):
        dist = distributions.NegBinom(n=10, p=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.06109619, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.5), 9, decimalPlace)
        self.assertAlmostEqual(dist.cdf(5), 0.1508789, decimalPlace)

    def test_ZTNegBinom(self):
        dist = distributions.ZTNegBinom(n=10, p=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.06115591, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.5), 10, decimalPlace)
        self.assertAlmostEqual(dist.cdf(5), 0.1500489, decimalPlace)

    # def test_ZMNegBinom(self):
    #     dist = distributions.ZMNegBinom(n=10, p=.5, p0m=.25)
    #     decimalPlace = 3
    #
    #     self.assertAlmostEqual(dist.pmf(5), 0.04586694, decimalPlace)
    #     self.assertAlmostEqual(dist.ppf(.5), 8, decimalPlace)
    #     self.assertAlmostEqual(dist.cdf(5), 0.3625367, decimalPlace)

    def test_Logser(self):
        dist = distributions.Logser(p=.5, p0m=.25)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.009016844, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.5), 1.0, decimalPlace)
        self.assertAlmostEqual(dist.cdf(5),  0.9933556, decimalPlace)

    def test_ZMLogser(self):
        dist = distributions.ZMLogser(p=.5,p0m=.25)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pmf(5), 0.006762633, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.5), 1.0, decimalPlace)
        self.assertAlmostEqual(dist.cdf(5),  0.9950167, decimalPlace)

class Test_Continuous_Distributions(unittest.TestCase):

    def test_Beta(self):
        dist = distributions.Beta(a=10, b=6)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 1.1436191094564112, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 0.6148493967089325, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.07692871333818019, decimalPlace)

    def test_Exponential(self):
        dist = distributions.Exponential(theta=.565)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.4381563, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 1.058119, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.2245021, decimalPlace)

    def test_Gamma(self):
        dist = distributions.Gamma(a=5, scale=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.0222292, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 2.203088, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.002344123, decimalPlace)

    def test_GenPareto(self):
        dist = distributions.GenPareto(c=.2, scale=1)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.596267326879216, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 0.6350460104896268, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.35006861370165454, decimalPlace)

    def test_Lognormal(self):
        dist = distributions.Lognormal(shape=1)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.6445272946093039, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 0.8819134589840192, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.21228796437921094, decimalPlace)

    def test_GenBeta(self):
        dist = distributions.GenBeta(shape1 = .4, shape2 = .5, shape3 = 6)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.5354495, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 0.8244533, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.1000981, decimalPlace)

    def test_Burr12(self):
        dist = distributions.Burr12(c=.5, d=.9)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.2529529209740369, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 0.8893595402048412, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.3699671280892439, decimalPlace)

    def test_Weibull(self):

        dist = distributions.Weibull(c=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.38109228105061776, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 0.3574090794724757, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.48871105232221823, decimalPlace)

    def test_InvWeibull(self):

        dist = distributions.InvWeibull(c=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.3730295569842341, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 1.568345663131631, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.2252122506990123, decimalPlace)

    def test_InvGamma(self):
        dist = distributions.InvGamma(a=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.2025384323986677, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 3.5047638201180478, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.03501498101966245, decimalPlace)

    def test_InvGauss(self):
        dist = distributions.InvGauss(mu=5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.5266136559318217, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 1.2598948382535147, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.1651782907399774, decimalPlace)

    def test_Fisk(self):
        dist = distributions.Fisk(c=.5)
        decimalPlace = 3

        self.assertAlmostEqual(dist.pdf(.45), 0.26699566652858964, decimalPlace)
        self.assertAlmostEqual(dist.ppf(.45), 0.6694214876033057, decimalPlace)
        self.assertAlmostEqual(dist.cdf(.45), 0.4014916240907944, decimalPlace)



if __name__ == '__main__':
    unittest.main()