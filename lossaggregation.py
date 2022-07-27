import time
import numpy as np
from scipy.special import factorial
from scipy.interpolate import interp1d
from . import config
from . import helperfunctions as hf

from twiggy import quick_setup, log


quick_setup()
logger = log.name('lossaggregation')


class LossAggregation:
    """
        Class representing the sum of random variables
        with a dependence structure specified by a copula and a set of given marginals.
    """

    def __init__(self, **kwargs):
        self.copula = kwargs['copula']
        self.copula_par = kwargs['copula_par']
        self.margins = kwargs['margins']
        self.margins_pars = kwargs['margins_pars']
        self.random_state = kwargs.get('random_state', int(time.time()))
        self.sample_size = kwargs.get('sample_size', 10000)
        self.__dist = self._dist_calculate()

    @property
    def random_state(self):
        return self.__random_state

    @random_state.setter
    def random_state(self, value):
        assert isinstance(value, (float, int)), logger.error('random_state has to be an integer')
        self.__random_state = int(value)

    @property
    def sample_size(self):
        return self.__sample_size

    @sample_size.setter
    def sample_size(self, value):
        assert isinstance(value, (float, int)), logger.error("sample_size has to be an integer")
        self.__sample_size = int(value)

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
            except Exception:
                logger.error('The marginal distribution %r is not parametrized correctly.' % j)
                raise
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
        except Exception:
            logger.error('Copula not correctly parametrized.\n See %s' % config.SITE_LINK)
            raise

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
        # Array of +1, -1, 0, indicating whether the new simpleces origined from sn must be summed,
        # subtracted or ignored, respectively.
        output = self.card.copy()
        greater = np.where(output > (1 / self.a))
        equal = np.where(output == (1 / self.a))
        lower = np.where(output < (1 / self.a))
        output[greater] = (-1) ** (self.d + 1 - output[greater])
        output[equal] = 0
        output[lower] = (-1) ** (1 + output[lower])
        return output

    def _private_prop_aep_initiate(self, x):
        self.__b = np.repeat(0, self.d).reshape(1, self.d)  # Vector b of the AEP algorithm.
        self.__h = np.array([[x]])  # Vector h of the AEP algorithm.
        self.__sn = np.array([1])  # Array of +1,-1, 0 indicating whether a volume must be summed,
        # subtracted or ignored, respectively.
        self.__vols = 0  # sum of 'volumes' * 'sn' used in AEP iteration

    def _private_prop_aep_delete(self):
        del self.__b
        del self.__h
        del self.__sn
        del self.__vols

    def _copula_rvs(self, size, random_state):
        result = config.COP_DICT[self.copula](**self.copula_par).rvs(size, random_state)
        return np.array(result)

    def _copula_cdf(self, k):
        result = config.COP_DICT[self.copula](**self.copula_par).cdf(k.transpose())
        return np.array(result)

    def _margins_ppf(self, k):
        result = [config.DIST_DICT[self.margins[j]](**self.margins_pars[j]).ppf(k[j, :]) for j in range(self.d)]
        return np.array(result)

    def _margins_cdf(self, k):
        result = [config.DIST_DICT[self.margins[j]](**self.margins_pars[j]).cdf(k[j, :]) for j in range(self.d)]
        return np.array(result)

    def _volume_calc(self):
        mat_ = np.expand_dims(self.mat, axis=2)
        h_ = self.a * self.__h
        b_ = np.expand_dims(self.__b.T, axis=0)
        s_ = self.s.reshape(-1, 1)
        v_ = np.hstack((b_ + h_ * mat_))  # np.dstack(zip( (b_ + h_*mat_) ))[0]    # <- WARNING
        c_ = self._copula_cdf(self._margins_cdf(v_)).reshape(-1, self.__b.shape[0])
        result = np.sum(c_ * (s_ * np.sign(h_) ** self.d), axis=0)
        return result

    def _sn_update(self):
        result = np.repeat(self.__sn, self.n_simpleces) * np.tile(self.m, self.__sn.shape[0])
        return result

    def _h_update(self):
        result = (1 - np.tile(self.card, len(self.__h)) * self.a) * np.repeat(self.__h, len(self.card))
        return result

    def _b_update(self):
        mat_ = self.mat[1:, :].transpose()
        h_ = np.repeat(self.__h, self.n_simpleces).reshape(-1, 1)
        times_ = int(h_.shape[0] / mat_.shape[1])
        result = np.repeat(self.__b, self.n_simpleces, 0)
        result = result + self.a * np.tile(h_, (1, self.d)) * np.tile(mat_, times_).transpose()
        return result

    def _aep(self, x, n_iter):
        self._private_prop_aep_initiate(x)
        cdf = self._volume_calc()[0]
        for _ in range(n_iter):
            self.__sn = self._sn_update()
            self.__b = self._b_update()
            self.__h = self._h_update()
            self.__vols = np.sum(self.__sn * self._volume_calc())
            cdf += self.__vols
        cdf += self.__vols * (self.ext - 1)
        self._private_prop_aep_delete()
        return cdf

    def _dist_calculate(self):
        u_ = self._copula_rvs(self.sample_size, self.random_state).T
        xsim = np.sum(self._margins_ppf(u_), axis=0)

        nodes, ecdf = hf.ecdf(xsim)
        epdf = np.repeat(1 / self.sample_size, self.sample_size)

        return {'epdf': epdf,
                'ecdf': ecdf,
                'nodes': nodes}

    def cdf(self, x, method="mc", **kwargs):
        """
        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :param method: method to approximate the cdf of the aggregate loss random variable
                        (i.e. the sum of random variables with a dependence structure specified by a copula).
                        One of AEP ('aep') and Monte Carlo simulation ('mc').
        :type method: ``string``
        :return: cumulative distribution function.
        :rtype: ``float``
        """
        if method not in config.LOSS_AGGREGATION_METHOD_LIST:
            raise ValueError("results: method must be one of %r." % config.LOSS_AGGREGATION_METHOD_LIST)

        x = np.ravel(x)
        output = np.empty(len(x))
        if method == 'aep':
            n_iter = kwargs.get('n_iter', 5)
            for i in range(len(output)):
                output[i] = self._aep(x[i], n_iter)
        elif method == 'mc':
            output = self._mc_cdf(x)

        return output

    def _mc_cdf(self, x):
        """
        Cumulative distribution function from Monte Carlo simulation.
        """
        x = np.ravel(x)
        y_ = np.append(0, self.__dist['nodes'])
        z_ = np.append(0, self.__dist['ecdf'])
        f = interp1d(y_, z_)
        x[x <= 0] = 0
        x[x >= self.__dist['nodes'][-1]] = self.__dist['nodes'][-1]
        return f(x)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function,
        inverse of cumulative distribution function from Monte Carlo simulation.
        """
        q = np.ravel(q)
        assert np.any(q <= 1), logger.error("q cannot exceed 1")
        assert np.any(q >= 0), logger.error("q cannot be less than 0")
        y_ = np.append(0, self.__dist['nodes'])
        z_ = np.append(0, self.__dist['ecdf'])
        f = interp1d(z_, y_)
        return f(q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :type n: ``int``
        :return: non-central moment of order n
        :rtype: ``float``
        """

        try:
            n = int(n)
        except Exception:
            logger.error('Please provide moment order "n" as an integer')
            raise

        return np.sum(self.__dist['nodes'] ** n * self.__dist['epdf'])

    def rvs(self, size=1, random_state=None):
        """
        Random variates. Based on Monte Carlo simulation.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        random_state = int(time.time()) if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size = int(size)
        except Exception:
            logger.error('Please provide size as an integer')
            raise

        np.random.seed(random_state)
        output = self.ppf(np.random.uniform(size=size))
        return output
