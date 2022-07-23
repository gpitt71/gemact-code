import numpy as np
from scipy.special import factorial
from . import config
from . import helperfunctions as hf

from twiggy import quick_setup, log


quick_setup()
logger = log.name('lossaggregation')


class LossAggregation:
    """
        Class representing the random variable of the sum of random variables
        with a dependence structure specified by a copula.
        Currently, it only computes the cdf using the AEP algorithm.
    """

    def __init__(self, **kwargs):

        # properties
        self.copula = kwargs['copula']
        self.copula_par = kwargs['copula_par']
        self.margins = kwargs['margins']
        self.margins_pars = kwargs['margins_pars']
        # private attributes
        self.__b = np.repeat(0, self.d).reshape(1, self.d)  # Vector b of the AEP algorithm.
        self.__h = None  # Vector h of the AEP algorithm.
        self.__sn = np.array(
            [1])  # Array of +1,-1, 0 indicating whether a volume must be summed, subtracted or ignored, respectively.
        self.__vols = 0  # sum of 'volumes' * 'sn' used in AEP iteration

    @property
    def margins_pars(self):
        return self.__margins_pars

    @margins_pars.setter
    def margins_pars(self, value):
        assert isinstance(value, list), logger.error("Please provide a list")
        assert len(value) == len(self.margins), logger.error("Margins and margins_pars must have the same dimension")

        for j in range(len(value)):
            assert isinstance(value[j], dict), logger.error("Please provide a list of dictionaries")

            try:
                config.DIST_DICT[self.margins[j]](**value[j])
            except:
                logger.error('The marginal distribution %r is not parametrized correctly.' % j)

        self.__margins_pars = value

    @property
    def margins(self):
        return self.__margins

    @margins.setter
    def margins(self, value):
        assert isinstance(value, list), logger.error("Please provide a list")

        assert len(value) <= config.DCEILING, logger.error("Number of dimensions exceeds limit of %r" % config.DCEILING)

        for j in range(len(value)):
            assert value[j] in config.DIST_DICT.keys(), '%r is not supported.\n See %s' % (value[j], config.SITE_LINK)
            assert 'severity' in config.DIST_DICT[value[j]].category(), logger.error(
                '%r is not a valid severity distribution' % value
            )
        self.__margins = value

    @property
    def copula(self):
        return self.__copula

    @copula.setter
    def copula(self, value):
        assert isinstance(value, str), logger.error('Copula name must be given as a string')
        assert value in config.COP_DICT.keys(), '%r copula is not supported.\n See %s' % (value, config.SITE_LINK)
        self.__copula = value

    @property
    def copula_par(self):
        return self.__copula_par

    @copula_par.setter
    def copula_par(self, value):
        assert isinstance(value, dict), 'The copula distribution parameters must be given as a dictionary'

        try:
            config.COP_DICT[self.copula](**value)
        except:
            logger.error('Copula not correctly parametrized.\n See %s' % config.SITE_LINK)

        self.__copula_par = value

    @property
    def d(self):
        return len(self.margins)

    @property
    def a(self):
        # Alpha parameter of the AEP algorithm.
        return 2. / (self.d + 1)

    @property
    def ext(self):
        # Probability correction of the AEP
        return ((self.d + 1) ** self.d) / (factorial(self.d) * 2 ** self.d)

    @property
    def mat(self):
        # Matrix of the vectors in the {0,1}**d space.
        return hf.cartesian_product(*([np.array([0, 1])] * self.d)).T

    @property
    def n_simpleces(self):
        # AEP number of new simpleces received in each step.
        return 2 ** self.d - 1

    @property
    def card(self):
        # AEP cardinality of the 'mat' matrix.
        return np.sum(self.mat, axis=1)[1:]

    @property
    def s(self):
        # AEP array of +1 or -1, indicating whether to sum or subtract a volume, respectively.
        return (-1) ** (self.d - np.sum(self.mat, axis=1))

    @property
    def m(self):
        # Array of +1, -1, 0, indicating whether the new simpleces origined from sn must be summed, subtracted or ignored, respectively.
        output = self.card.copy()
        greater = np.where(output > (1 / self.a))
        equal = np.where(output == (1 / self.a))
        lower = np.where(output < (1 / self.a))
        output[greater] = (-1) ** (self.d + 1 - output[greater])
        output[equal] = 0
        output[lower] = (-1) ** (1 + output[lower])
        return output

    def copula_cdf(self, k):
        result = config.COP_DICT[self.copula](**self.copula_par).cdf(k.transpose())
        return np.array(result)

    def margins_cdf(self, k):
        result = [config.DIST_DICT[self.margins[j]](**self.margins_pars[j]).cdf(k[j, :]) for j in range(self.d)]
        return np.array(result)

    def volume_calc(self):
        mat_ = np.expand_dims(self.mat, axis=2)
        h_ = self.a * self.__h
        b_ = np.expand_dims(self.__b.T, axis=0)
        s_ = self.s.reshape(-1, 1)
        v_ = np.hstack((b_ + h_ * mat_))  # np.dstack(zip( (b_ + h_*mat_) ))[0]    # <- WARNING
        c_ = self.copula_cdf(self.margins_cdf(v_)).reshape(-1, self.__b.shape[0])
        result = np.sum(c_ * (s_ * np.sign(h_) ** self.d), axis=0)
        return result

    def sn_update(self):
        result = np.repeat(self.__sn, self.n_simpleces) * np.tile(self.m, self.__sn.shape[0])
        return result

    def h_update(self):
        result = (1 - np.tile(self.card, len(self.__h)) * self.a) * np.repeat(self.__h, len(self.card))
        return result

    def b_update(self):
        mat_ = self.mat[1:, :].transpose()
        h_ = np.repeat(self.__h, self.n_simpleces).reshape(-1, 1)
        times_ = int(h_.shape[0] / mat_.shape[1])
        result = np.repeat(self.__b, self.n_simpleces, 0)
        result = result + self.a * np.tile(h_, (1, self.d)) * np.tile(mat_, times_).transpose()
        return result

    def _AEP(self, k, n_iter):

        self.__h = np.array([[k]])
        prob = self.volume_calc()[0]
        for _ in range(n_iter):
            self.__sn = self.sn_update()
            self.__b = self.b_update()
            self.__h = self.h_update()
            self.__vols = np.sum(self.__sn * self.volume_calc())
            prob += self.__vols
        return prob + self.__vols * (self.ext - 1)

    def _mc_approximation(self):
        print("Currently not available. Work in progress.")
        return 0

    def cdf(self, k, method="aep", **kwargs):
        output = None
        if method not in config.LOSS_AGGREGATION_METHOD_LIST:
            raise ValueError("results: method must be one of %r." % config.LOSS_AGGREGATION_METHOD_LIST)

        if method == 'aep':
            n_iter = kwargs.get('n_iter', 5)
            output = self._AEP(k=k, n_iter=n_iter)
        elif method == 'mc':
            output = self._mc_approximation()

        return output