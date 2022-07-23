#from scipy import stats,special
import numpy as np
import scipy.stats
import scipy.special
import time # used for setting random number generator seed if None
from twiggy import quick_setup, log
from . import helperfunctions as hf

quick_setup()
logger= log.name('distributions')

## Distribution 
class _Distribution():
    """
    Class representing a probability distribution. To be inherited.
    """

    def __init__(self):
        pass
    
    def rvs(self, size=1, random_state=None, **kwargs):

        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator (default=None).
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = int(time.time()) if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')

        return self.dist.rvs(size=size, random_state=random_state, **kwargs)

    def cdf(self, x, **kwargs):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.dist.cdf(x, **kwargs)

    def logcdf(self, x, **kwargs):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.dist.logcdf(x, **kwargs)

    def sf(self, x, **kwargs):
        """
        Survival function, 1 - cumulative distribution function.

        :param x: quantile where the survival function is evaluated.
        :type x: ``int``
        :return: survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.dist.sf(x, **kwargs)

    def logsf(self, x, **kwargs):
        """
        Natural logarithm of the survival function.

        :param x: quantile where the logarithm of the survival function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.dist.logsf(x, **kwargs)

    def ppf(self, q, **kwargs):
        """
        Percent point function, a.k.a. the quantile function, inverse of the cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        return self.dist.ppf(q=q, **kwargs)

    def isf(self, q, **kwargs):
        """
        Inverse survival function (inverse of sf).

        :param q: level at which the inverse survival function is evaluated.
        :type q: ``float``
        :return: inverse survival function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        return self.dist.ppf(1 - q, **kwargs)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’) function.

        :param moments: moments to be returned.
        :type moments: string, optional
        :return: moments.
        :rtype: tuple
        """
        return self.dist.stats(moments=moments)

    def entropy(self, **kwargs):
        """
        (Differential) entropy of the random variable.

        :return: entropy
        :rtype: ``numpy.ndarray``

        """
        return self.dist.entropy(**kwargs)

    def expect(self, func, lb=None, ub=None, conditional=False):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: function for which integral is calculated. Takes only one argument. The default is the identity mapping f(x) = x.
        :type func: ``callable``, optional
        :param lb: Lower bound for integration. Default is set to the support of the distribution.
        :type lb: ``float``, optional
        :param ub: Upper bound for integration. Default is set to the support of the distribution.
        :type ub: ``float``, optional
        :param conditional: If True, the integral is corrected by the conditional probability of the integration interval.
        The return value is the expectation of the function, conditional on being in the given interval. Default is False.
        :type conditional: ``bool``, optional
        :return: the calculated expected value.
        :rtype: ``float``

        """
        return self.dist.expect(func, lb=lb, ub=ub, conditional=conditional)

    def median(self, **kwargs):
        """
        Median of the distribution.

        :return: median.
        :rtype: ``float``

        """
        return self.dist.median(**kwargs)

    def mean(self, **kwargs):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``float``

        """
        return self.dist.mean(**kwargs)

    def var(self, **kwargs):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``float``

        """
        return self.dist.var(**kwargs)

    def std(self, **kwargs):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``float``

        """
        return self.dist.std(**kwargs)

    def interval(self, **kwargs):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :type alpha: ``float``
        :return: Endpoints
        :rtype: tuple

        """
        return self.dist.interval(**kwargs)
    
    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :type n: ``int``
        :return: non-central moment of order n
        :rtype: ``float``
        """
        return self.dist.moment(n=n)

## Discrete distribution 
class _DiscreteDistribution(_Distribution):
    """
    Class representing a discrete probability distribution. To be inherited.
    Child class of ``_Distribution`` class.
    """

    def __init__(self):
        super().__init__()
        pass
    
    @staticmethod
    def category():
        return {'frequency'}

    def pmf(self, x, **kwargs):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.dist.pmf(x, **kwargs)

    def logpmf(self, x, **kwargs):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``
        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.dist.logpmf(x, **kwargs)
    
    def abk(self):
        """
        Function returning (a, b, x) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        try:
            return self.a, self.b, self.p0

        except ValueError:
            print("Distribution without (a, b, x) parametrization")

## Continuous distribution 
class _ContinuousDistribution(_Distribution):
    """
    Class representing a continuous probability distribution. To be inherited.
    Child class of ``_Distribution`` class.
    """

    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def category():
        return {'severity'}

    def pdf(self, x, **kwargs):
        """
        Probability density function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``float``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.pdf(x, **kwargs)

    def logpdf(self, x, **kwargs):
        """
        Natural logarithm of the probability denisty function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``float``
        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.dist.logpdf(x, **kwargs)

    def fit(self, data, **kwargs):
        """
        Return estimates of shape (if applicable), location, and scale parameters from data.
        The default estimation method is Maximum Likelihood Estimation (MLE), but Method of Moments (MM) is also available.
        Refer to ``scipy.stats.rv_continuous.fit``

        :param data: data to use in estimating the distribution parameters.
        :type x: array_like
        :return: parameter_tuple. Estimates for any shape parameters (if applicable), followed by those for location and scale.
        :rtype: tuple of floats

        """
        return self.dist.fit(data, **kwargs)

## Poisson
class Poisson(_DiscreteDistribution):
    """
    Poisson distribution.
    Wrapper to scipy poisson distribution (``scipy.stats._discrete_distns.poisson_gen``)
    Refer to :py:class:'~_DiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          Poisson distribution parameter mu (rate).

    """

    def __init__(self, loc=0, **kwargs):
        super().__init__()
        self.mu = kwargs['mu']
        self.loc = loc

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        assert (value > 0), logger.error("mu has to be > 0")
        self.__mu = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.poisson(mu=self.mu, loc=self.loc)

    @property
    def a(self):
        return 0
    
    @property
    def b(self):
        return self.mu

    @property
    def p0(self):
        return np.exp(-self.mu)

    @staticmethod
    def name():
        return 'poisson'

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated.
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return np.exp(self.b * (f - 1))
    
    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.mu = nu * self.mu
        return

## Binomial
class Binom(_DiscreteDistribution):
    """
    Binomial distribution.
    Wrapper to scipy binomial distribution (``scipy.stats._discrete_distns.binom_gen``).
    Refer to :py:class:'~_DiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *n* (``int``) --
          Number of trials.
        * *p* (``float``) --
          Probability of a success, parameter of the binomial distribution.

    """

    def __init__(self, loc=0, **kwargs):
        super().__init__()
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.loc = loc
    
    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert (isinstance(value, int) and (value > 0)),\
             logger.error("n has to be a positive integer")
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert (0 <= value <= 1), logger.error("p must be in [0, 1].")
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.binom(n=self.n, p=self.p, loc=self.loc)

    @property
    def a(self):
        return -self.p / (1 - self.p)
    
    @property
    def b(self):
        return (self.n + 1) * (self.p/(1 - self.p))
    
    @property
    def p0(self):
        return (1 - self.p) ** self.n

    @staticmethod
    def name():
        return 'binom'

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (1 + self.a / (self.a - 1)*(f - 1))**(-self.b/self.a-1)
    
    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = nu*self.p
        return

## Geometric
class Geom(_DiscreteDistribution):
    """
    Geometric distribution.
    Wrapper to scipy geometric distribution (``scipy.stats._discrete_distns.geom_gen``).
    Refer to :py:class:'~_DiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *p* (``float``) --
          Probability parameter of the geometric distribution.

    """

    def __init__(self, loc=0, **kwargs):
        super().__init__()
        self.p = kwargs['p']
        self.loc = loc

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert (0 <= value <= 1), logger.error("p must be in [0, 1].")
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.geom(p=self.p, loc=self.loc)

    @property
    def a(self):
        return 1 - self.p
    
    @property
    def b(self):
        return 0
    
    @property
    def p0(self):
        return np.array([((1 - self.p)/self.p)**(-1)])

    @staticmethod
    def name():
        return 'geom'

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (1- self.a/(1-self.a) * (f-1))**(-1)

    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1/(1 + nu*(1-self.p)/self.p)
        return

## Negative Binomial
class NegBinom(_DiscreteDistribution):
    """
    Negative Binomial distribution.
    Wrapper to scipy negative binomial distribution (``scipy.stats._discrete_distns.nbinom_gen``).
    Refer to :py:class:'~_DiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *n* (``int``) --
          Size parameter of the negative binomial distribution.
        * *p* (``float``) --
          Probability parameter of the negative binomial distribution.

    """

    def __init__(self, loc=0, **kwargs):
        super().__init__()
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.loc = loc
    
    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert (isinstance(value, int) and (value > 0)),\
             logger.error("n has to be a positive integer")
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert (0 <= value <= 1), logger.error("p must be in [0, 1].")
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return self.scipy.stats.nbinom(n=self.n, p=self.p, loc=self.loc)

    @property
    def a(self):
        return 1 - self.p
    
    @property
    def b(self):
        return self.__b
    
    @property
    def p0(self):
        return (self.n - 1)*(1 - self.p)

    @staticmethod
    def name():
        return 'nbinom'

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated.
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (1- self.a/(1-self.a) * (f-1))**(-self.b/self.a - 1)

    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1 / (1 + nu*(1-self.p)/self.p)
        return

## Zero-truncated Poisson
class ZTPoisson():
    """
    Zero-truncated Poisson distribution.
    Poisson distribution with no mass (truncated) in 0.
    scipy reference non-zero-truncated distribution: ``scipy.stats._discrete_distns.poisson_gen``

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          Zero-truncated Poisson distribution parameter mu (rate).

    """

    def __init__(self, loc=0, **kwargs):
        self.mu = kwargs['mu']
        self.loc = loc

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        assert value > 0, logger.error('mu has to be positive')
        self.__mu = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.poisson(mu=self.mu, loc=self.loc)

    @property
    def a(self):
        return 0
    
    @property
    def b(self):
        return self.mu
    
    @property
    def p0(self):
        return np.exp(-self.mu)

    @staticmethod
    def category():
        return {'frequency', 'zt'}

    @staticmethod
    def name():
        return 'ztpoisson'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        temp = self.dist.pmf(x)/(1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.0
            else:
                temp[zeros] = 0.0
            return temp
    
    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """

        return (self.dist.cdf(x) - self.dist.cdf(0))/(1 - self.dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        random_state = int(time.time()) if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')
        
        np.random.seed(random_state)
        q_ = np.random.uniform(low=self.dist.cdf(0), high=1, size=size)
        return self.dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        q_ = np.array(q)
        return (self.dist.ppf(q=q_*(1 - self.dist.cdf(0)) + self.dist.cdf(0)))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (np.exp(self.b*f) - 1)/(np.exp(self.b) - 1)

    def abk(self):
        """
        Function returning (a, b, x) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, 0
    
    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.mu = nu * self.mu
        return

## Zero-modified Poisson
class ZMPoisson():
    """
    Zero-modified Poisson distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified Poisson distribution.
    scipy reference non-zero-modified distribution: ``scipy.stats._discrete_distns.poisson_gen``

    :param loc: location parameter (default=0).
    :type loc: ``int``, optional 
    :param maxDiff: threshold to determine which method to generate random variates (default=0.95).
    :type maxDiff: ``float``, optional 
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          Zero-modified Poisson distribution rate parameter.
        * *p0M* (``numpy.float64``) --
          Zero-modified Poisson mixing parameter. Resulting probability mass in zero.

    """

    def __init__(self, loc=0, maxDiff = 0.95, **kwargs):
        self.loc = loc
        self.mu = kwargs['mu']
        self.p0M = kwargs['p0M']
        self.maxDiff = maxDiff

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error('loc has to be int type')
        self.__loc = value

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        assert value > 0, logger.error('mu must be positive')
        self.__mu = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert (0 <= value <= 1), logger.error('p0M must be in [0, 1].')
        self.__p0M = value

    @property
    def maxDiff(self):
        return self.__maxDiff

    @maxDiff.setter
    def maxDiff(self, value):
        self.__maxDiff = value

    @property
    def dist(self):
        return scipy.stats.poisson(mu=self.mu, loc=self.loc)

    @property
    def a(self):
        return 0
    
    @property
    def b(self):
        return self.mu
    
    @property
    def p0(self):
        return np.exp(-self.mu)

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmpoisson'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        
        temp = self.dist.pmf(x)*(1 - self.p0M)/(1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = self.p0M
            else:
                temp[zeros] = self.p0M
            return temp

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.p0M + (1 - self.p0M)*(self.dist.cdf(x) - self.dist.cdf(0))/(1 - self.dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = int(time.time()) if (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        if (self.mu == 0):
            u_ = np.random.uniform(0, 1, size)
            idx = u_ <= self.p0M
            u_[idx] = 0
            u_[np.invert(idx)] = 1
            return u_
        
        if (self.p0M >= self.p0):
            u_ = np.random.uniform(0, (1 - self.p0), size)
            idx = (u_ <= (1 - self.p0M))
            u_[idx] = self.dist.rvs(mu=self.mu, size=np.sum(idx))
            u_[np.invert(idx)] = 0
            return u_
       
        if ((self.p0 - self.p0M) < self.maxDiff):
            # rejection method
            u_ = []
            while len(u_) < size:
                x_ = self.dist.rvs(1, self.mu)
                if (x_ != 0 or np.random.uniform(0, self.p0*(1 - self.p0M), 1) <= (1 - self.p0)*self.p0M):
                    u_.append(x_)
            return np.asarray(u_)
        else:
            # inversion method
            u_ = np.random.uniform((self.p0 - self.p0M)/(1 - self.p0M), 1, size)
            return self.dist.ppf(u_, self.mu)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumultaive distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        
        q_ = np.array(q)
        temp = self.dist.ppf((1 - self.dist.cdf(0))*(q_ - self.p0M)/(1 - self.p0M)+self.dist.cdf(0))
        return temp

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return self.p0M+(1 - self.p0M)*(np.exp(self.b*f) - 1)/(np.exp(self.b) - 1)

    def abk(self):
        """
        Function returning (a, b, x) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0M 
    
    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p0M = (self.p0M - np.exp(-self.mu) +\
            np.exp(-nu * self.mu) - self.p0M * np.exp(-nu * self.mu))/\
                (1-np.exp(-self.mu))
        self.mu = nu * self.mu
        return

## Zero-truncated binomial
class ZTBinom():
    """
    Zero-truncated binomial distribution. Binomial distribution with no mass (truncated) in 0.
    scipy reference non-zero-truncated distribution: ``scipy.stats._discrete_distns.binom_gen``.

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Zero-trincated binomial distribution size parameter n.
        * *p*(``float``) --
          Zero-truncated binomial distribution probability parameter p.

    """

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert (isinstance(value, int) and (value > 0)),\
             logger.error("n has to be a positive integer")
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert (0 <= value <= 1), logger.error("p must be in [0, 1].")
        self.__p = value

    @property
    def dist(self):
        return scipy.stats.binom(n=self.n, p=self.p)

    @property
    def a(self):
        return -self.p/(1 - self.p)
    
    @property
    def b(self):
        return (self.n + 1)*(self.p/(1 - self.p))
    
    @property
    def p0(self):
        return (1 - self.p)**self.n

    @staticmethod
    def category():
        return {'frequency', 'zt'}

    @staticmethod
    def name():
        return 'ztbinom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        
        temp = self.dist.pmf(x)/(1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return (self.dist.cdf(x) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = int(time.time()) if  (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        q_ = np.random.uniform(low=self.dist.cdf(0), high=1, size=size)
        return self.dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        q_ = np.array(q)
        return self.dist.ppf(q=q_ * (1 - self.dist.cdf(0)) + self.dist.cdf(0))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        a_ = self.a
        b_ = self.b
        return ((1+a_/(a_ - 1)*(f - 1))**(-b_/a_-1)-(1 - a_)**(b_/a_+1))/(1-(1 - a_)**(b_/a_+1))

    def abk(self):
        """
        Function returning (a, b, x) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, 0

    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = nu*self.p
        return

## Zero-modified binomial
class ZMBinom():
    """
    Zero-modified binomial distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified binomial distribution.
    scipy reference non-zero-modified distribution: ``scipy.stats._discrete_distns.binom_gen``.

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``numpy.float64``) --
          Zero-modified binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          Zero-modified binomial distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          Zero-modified binomial mixing parameter.
          
    """

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.p0M = kwargs['p0M']

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert (isinstance(value, int) and value >= 1), logger.error('n must be a natural number')
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert (0 <= value <= 1), logger.error('p must be in [0, 1].')
        self.__p = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert (0 <= value <= 1), logger.error('p0M must be in [0, 1].')
        self.__p0M = value

    @property
    def dist(self):
        return scipy.stats.binom(n=self.n, p=self.p)

    @property
    def distZT(self):
        return ZTBinom(n=self.n, p=self.p)

    @property
    def a(self):
        return -self.p / (1 - self.p)
    
    @property
    def b(self):
        return (self.n + 1) * (self.p / (1 - self.p))
    
    @property
    def p0(self):
        return np.array([(1 - self.p) ** self.n])

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmbinom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.pmf(x)*(1 - self.p0M)/(1 - self.p0)

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.p0M + (1 - self.p0M)*(self.dist.cdf(x) - self.dist.cdf(0))/(1 - self.dist.cdf(0))

    def logcdf(self, x):
        """
        Natural lograithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = int(time.time()) if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        r_ = scipy.stats.bernoulli(p=1-self.p0M).rvs(size, random_state=random_state)
        c_ = np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_]= self.distZT.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        q_ = np.array(q)
        temp = self.dist.ppf((1 - self.dist.cdf(0)) * (q_ - self.p0M) / (1 - self.p0M) + self.dist.cdf(0))
        return temp

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        a_ = self.a
        b_ = self.b
        return self.p0M+(1 - self.p0M)*((1 + a_/(a_ - 1)*(f - 1))**(-b_/a_ - 1)\
            - (1 - a_)**(b_/a_ + 1))/(1 - (1 - a_)**(b_/a_ + 1))

    def abk(self):
        """
        Function returning (a, b, x) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0M

    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p0M = (self.p0M-(1-self.p)**self.n+ \
            (1-nu*self.p)**self.n-self.p0M*(1-nu*self.p)**self.n)/ \
                (1-(1-self.p)**self.n)
        self.p = nu*self.p
        return

## Zero-truncated geometric
class ZTGeom():
    """
    Zero-truncated geometric distribution. Geometric distribution with no mass (truncated) in 0.
    scipy reference non-zero-truncated distribution: ``scipy.stats._discrete_distns.geom_gen``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          Zero-truncated geometric distribution probability parameter p.

    """
    def __init__(self, **kwargs):
        self.p = kwargs['p']

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p must be in [0, 1].')
        self.__p = value

    @property
    def dist(self):
        return scipy.stats.geom(p=self.p)

    @property
    def a(self):
        return 1 - self.p
    
    @property
    def b(self):
        return 0
    
    @property
    def p0(self):
        return self.p

    @staticmethod
    def category():
        return {'frequency', 'zt'}

    @staticmethod
    def name():
        return 'ztgeom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        temp = (self.dist.pmf(x + 1)) / (1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return (self.dist.cdf(x) - self.dist.cdf(0))/(1 - self.dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = int(time.time()) if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')
        
        q_ = np.random.uniform(low=self.dist.cdf(0), high=1, size=size)
        return self.dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        q_ = np.array(q)
        return (self.dist.ppf(q=q_ * (1 - self.dist.cdf(0)) + self.dist.cdf(0)))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (1 / (1 - (f - 1) / (1 - self.a)) - 1 + self.a) / self.a

    def abk(self):
        """
        Function returning (a, b, x) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, 0

    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1/(1 + nu*(1-self.p)/self.p)
        return

## Zero-modified geometric
class ZMGeom():
    """
    Zero-modified geometric distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified geometric distribution.
    scipy reference non-zero-modified distribution: `scipy.stats._discrete_distns.geom_gen``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          Zero-modified geometric distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          Zero-modified geometric mixing parameter.
    """

    def __init__(self, **kwargs):
        self.p = kwargs['p']
        self.p0M = kwargs['p0M']

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            "p must be in [0, 1].")
        self.__p = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p0M must be in [0, 1].')
        self.__p0M = value

    @staticmethod
    def category():
        return {'frequency'}

    @property
    def a(self):
        return 1 - self.p
    
    @property
    def b(self):
        return 0

    @property
    def p0(self):
        return self.p

    @property
    def dist(self):
        return scipy.stats.geom(p=self.p)

    @property
    def distZT(self):
        return ZTGeom(p=self.p)

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmgeom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self.dist.pmf(x) + 1) * (1 - self.p0M) / (1 - self.p0)

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.p0M + (1 - self.p0M) * (self.dist.cdf(x) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        r_ = scipy.stats.bernoulli(p=1 - self.p0M).rvs(size, random_state=random_state)
        c_ = np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_] = self.distZT.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        q_ = np.array(q)
        return self.dist.ppf((1 - self.dist.cdf(0)) * (q_ - self.p0M) / (1 - self.p0M) + self.dist.cdf(0))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return self.p0M+(1 - self.p0M)*(1/(1 - (f - 1)/(1 - self.a)) - 1 + self.a)/self.a

    def abk(self):
        """
        Function returning (a, b, x) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0M

    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        beta = (1-self.p)/self.p
        self.p0M = (self.p0M-(1+beta)**-1 + (1+nu*beta)**-1 - self.p0M*(1+nu*beta)**-1) / (1-(1+beta)**-1)
        self.p = 1/(1+nu*beta)
        return

## Zero-truncated negative binomial
class ZTNegBinom():
    """
    Zero-truncated negative binomial distribution. Negative binomial distribution with no mass (truncated) in 0.
    scipy reference non-zero-truncated distribution: ``scipy.stats._discrete_distns.nbinom_gen``.

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Zero-truncated negative binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          Zero-truncated negative binomial distribution probability parameter p.
    """

    name = 'ztnbinom'
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert isinstance(value, int) and value >= 0, logger.error(
            'n must be a positive number')
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p must be in [0, 1].')
        self.__p = value

    @property
    def a(self):
        return 1 - self.p
    
    @property
    def b(self):
        return (self.n - 1) * (1 - self.p)

    @property
    def p0(self):
        return self.p ** self.n

    @property
    def dist(self):
        return scipy.stats.nbinom(n=self.n, p=self.p)

    @staticmethod
    def category():
        return {'frequency', 'zt'}

    @staticmethod
    def name():
        return 'ztnbinom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        temp = (self.dist.pmf(x))/(1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return (self.dist.cdf(x) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = int(time.time()) if (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        q_ = np.random.uniform(low=self.dist.cdf(0), high=1, size=size)
        return self.dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        q_ = (np.array(q) * (1 - self.dist.cdf(0)) + self.dist.cdf(0))
        return self.dist.ppf(q=q_)

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        c_ = self.b/self.a+1
        d_ = 1-self.a
        return ((1/(1-(f-1)*self.a/d_))**c_-d_**c_)/(1-d_**c_)

    def abk(self):
        """
        Function returning (a, b, x) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, 0
    
    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1 / (1 + nu*(1-self.p)/self.p)
        return

## Zero-modified negative binomial
class ZMNegBinom():
    """
    Zero-modified negative binomial distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified negative binomial distribution.
    scipy reference non-zero-modified distribution: ``scipy.stats._discrete_distns.nbinom_gen``.

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Zero-truncated negative binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          Zero-modified negative binomial distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          Zero-modified negative binomial mixing parameter.

    """

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.p0M = kwargs['p0M']

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert isinstance(value, int) and value >= 0, logger.error(
            'n must be a positive number')
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p must be in [0, 1].')
        self.__p = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p0M must be in [0, 1].')
        self.__p0M = value

    @property
    def a(self):
        return 1 - self.p
    
    @property
    def b(self):
        return (self.n - 1) * (1 - self.p)

    @property
    def p0(self):
        return np.array([(1 / self.p) ** -self.n])

    @property
    def dist(self):
        return scipy.stats.nbinom(n=self.n, p=self.p)

    @property
    def distZT(self):
        return ZTNegBinom(n=self.n, p=self.p)

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmnbinom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self.dist.pmf(x) * (1 - self.p0M)) / (1 - self.p0)

    def logpmf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.p0M + (1 - self.p0M) * (self.dist.cdf(x) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = int(time.time()) if (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        r_= scipy.stats.bernoulli(p=1-self.p0M).rvs(size, random_state=random_state)
        c_= np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_] = self.distZT.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        p_ = np.array(q)
        return self.dist.ppf((1 - self.dist.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.dist.cdf(0))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, x) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        c_ = 1-self.a
        d_ = (self.b/self.a+1)

        return self.p0M+(1-self.p0M)*((1/(1-(f-1)*self.a/c_))**d_-c_**d_)/(1-c_**d_)

    def abk(self):
        """
        Function returning (a, b, x) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0M

    def par_franchise_adjuster(self, nu):
        """
        Parameter correction in case of deductible (franchise).
        
        :param nu: severity model survival function at the deductible (franchise)
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        beta = (1-self.p)/self.p
        self.p0M = (self.p0M-(1+beta)**(-self.n) + (1+nu*beta)**-self.n - self.p0M * (1+nu*beta)**-self.n)/ \
            (1-(1+beta)**-self.n)
        self.p = 1/(1+nu*beta)
        return

## Zero-modified discrete logarithmic
class ZMLogser():
    """
    Zero-modified (discrete) logarithmic (log-series, series) distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified logarithmic distribution.
    scipy reference non-zero-modified distribution: ``scipy.stats._discrete_distns.logser_gen``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          ZM discrete logarithmic distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          ZM discrete logarithmic mixing parameter.
    """

    def __init__(self, **kwargs):
        self.p = kwargs['p']
        self.p0M = kwargs['p0M']

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p must be in [0, 1].')
        self.__p = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p0M must be in [0, 1].')
        self.__p0M = value

    @property
    def p0(self):
        return 0

    @property
    def dist(self):
        return scipy.stats.logser(p=self.p)

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmlogser'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self.dist.pmf(x) * (1 - self.p0M)) / (1 - self.p0)

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.p0M + (1 - self.p0M) * (self.dist.cdf(x) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))
   
    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = int(time.time()) if (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        q_ = np.random.uniform(0, 1, size)
        return self.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        p_ = np.array(q)
        temp = self.dist.ppf((1 - self.dist.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.dist.cdf(0))

        zeros = np.where(p_ <= self.p0M)[0]
        if zeros.size == 0:
            return temp
        else:
            if p_.shape == ():
                temp = self.p0M
            else:
                temp[zeros] = self.p0M
            return temp

## Beta
class Beta(_ContinuousDistribution):
    """
    Wrapper to scipy beta distribution.
    ``scipy.stats._continuous_distns.beta_gen``

    :param scale: beta scale parameter.
    :type scale: ``float``
    :param loc: beta location parameter.
    :type loc: ``float``
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *a* (``int`` or ``float``) --
          shape parameter a.
        * *b* (``int`` or ``float``) --
          shape parameter b.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        super().__init__()
        self.a = kwargs['a']
        self.b = kwargs['b']
        self.scale = scale
        self.loc = loc

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'a must be a positive number')
        self.__a = value

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'b must be a positive number')
        self.__b = value

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        assert value > 0, logger.error('mu has to be positive')
        self.__mu = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.beta(
            a=self.a,
            b=self.b,
            loc=self.loc,
            scale=self.scale
            )
    @staticmethod
    def name():
        return 'beta'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v]).flatten()
        output = v.copy()
        u = v/self.scale
        output[v > 0] = v[v > 0]*\
            (1-scipy.special.betaincinv(self.a, self.b, u[v > 0])) +\
                scipy.special.betaincinv(self.a+1, self.b, u[v > 0])*\
                    self.scale*scipy.special.gamma(self.a+self.b) *\
                        scipy.special.gamma(self.a+1)/(scipy.special.gamma(self.a+self.b+1)*\
                            scipy.special.gamma(self.a))
        return output


    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        loc = (loc - self.loc) 
        # return 1 - scipy.stats.beta(loc=0,a=self.a,b=self.b,scale=self.scale).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)

## Exponential
class Exponential(_ContinuousDistribution):
    """
    Expontential distribution.
    scipy reference distribution: ``scipy.stats._continuous_distns.expon_gen``
    
    :param theta: exponential distribution theta parameter.
    :type theta: ``float``
    :param loc: location parameter
    :type loc: ``float``
    """

    def __init__(self, loc=0, theta=1):
        super().__init__()
        self.theta = theta
        self.loc = loc
        # self.dist: read only property -- see below

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, value):
        assert value > 0 , logger.error(
            'theta must be positive')
        self.__theta = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error('loc must be float or int type')
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.expon(loc=self.loc)

    @staticmethod
    def name():
        return 'exponential'

    def pdf(self, x):
        """
        Probability density function.

        :param x: quantile where probability density function is evaluated.
        :type x: ``numpy.ndarray``
        :return: probability density function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.theta * self.dist.pdf(self.loc + self.theta*x)

    def logpdf(self, x):
        """
        Natural logarithm of the probability density function.

        :param x: the log of the probability function will be computed in x.
        :type x:``numpy.ndarray``
        :return: logpdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.logpdf(self.theta*x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: the cumulative distribution function will be computed in x.
        :type x: ``numpy.ndarray``
        :return: cumulative distribution function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.cdf(self.loc+self.theta * (x-self.loc))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: point where the natural logarithm of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.logcdf(self.theta*x)

    def sf(self, x):
        """
        Survival function, 1 - cumulative distribution function.

        :param x: point where the survival is evaluated.
        :type x: ``int``
        :return: survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.sf(self.theta*x)

    def logsf(self, x):
        """
        Natural logarithm of the survival function.

        :param x: point where the natural logarithm of the survival function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.logsf(self.theta*x)

    def isf(self, x):
        """
        Inverse survival function (inverse of sf).

        :param x: point where the inverse of the survival function is evaluated.
        :type x: ``numpy.ndarray``
        :return: inverse of the survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.isf(self.theta*x)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

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
            size=int(size)
        except:
            logger.error('Please provide size as an integer')
        
        return scipy.stats.expon.rvs(size=size, random_state=random_state)/self.theta + self.loc

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: (differential) entropy.
        :rtype: ``numpy.float64``
        """
        return 1-np.log(self.theta)

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return 1 / self.theta

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return 1/ self.theta**2

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return np.sqrt(self.variance)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of the cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """

        try:
            q=np.array(q)
            assert isinstance(q, np.ndarray), logger.error('q values must be an array')
        except:
            logger.error('Please provide the x quantiles you want to evaluate as an array')

        temp= -np.log(1-q)/self.theta

        zeros = np.where(((q >= 1.) & (q <=0.)))[0]

        if zeros.size == 0:
            return temp
        else:
            if q.shape == ():
                temp = np.float64('nan')
            else:
                temp[zeros] = np.float64('nan')
            return temp

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v])
        out = (1 - np.exp(-self.theta * v)) / self.theta
        out[v < 0] = v[v < 0]
        return out

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        # return (1 - scipy.stats.expon.cdf(self.theta * (d - loc)))
        return 1 - self.dist.cdf(self.theta*(low - loc) + self.loc)
        
## Gamma
class Gamma(_ContinuousDistribution):
    """
    Gamma distribution.
    When a is an integer it reduces to an Erlang distribution.
    When a=1 it reduces to an Exponential distribution.
    Wrapper to scipy gamma distribution (``scipy.stats._continuous_distns.gamma_gen ``).

    :param scale: scale parameter (inverse of the rate parameter).
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *a* (``int`` or ``float``) --
          shape parameter a.

    """

    def __init__(self, loc=0, scale=1, **kwargs):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.a = kwargs['a']

    @property
    def a(self):
        return self.__a
    
    @a.setter
    def a(self, value):
        assert isinstance(value, (float, int)), logger.error("a has to be float or int type")
        self.__a = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error("loc has to be float type")
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)), logger.error("scale has to be float or int type")
        self.__scale = value

    @property
    def dist(self):
        return scipy.stats.gamma(a=self.a, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'gamma'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v])
        try:
            beta = 1 / self.scale
        except:
            beta = 1
        alpha = self.a
        out = (alpha / beta) * scipy.special.gammainc(alpha + 1, beta * v) + v * (
                1 - scipy.special.gammainc(alpha, beta * v))
        out[v < 0] = v[v < 0]
        return out

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            beta = 1 / self.scale
        except:
            beta = 1

        self.scale = 1/beta
        loc = (loc - self.loc) 
        # return 1 - scipy.stats.gamma(loc=0, a=self.a, scale=1/beta).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)

## Generalized Pareto
class GenPareto(_ContinuousDistribution):
    """
    Wrapper to scipy genpareto distribution.
    When c=0 it reduces to an Exponential distribution.
    When c=-1 it reduces to a uniform distribution.
    When the correct parametrization is adopted, it is possible to fit all the Pareto types.
    scipy reference distribution: ``scipy.stats._continuous_distns.genpareto_gen ``

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc:``float``
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter c.

    """
 
    def __init__(self, loc=0, scale=1, **kwargs):
        super().__init__()
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc

    @property
    def c(self):
        return self.__c
    
    @c.setter
    def c(self, value):
        assert isinstance(value, (float, int)), logger.error("c has to be float or int type")
        self.__c = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)), logger.error("scale has to be float type")
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error("loc has to be float or int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.genpareto(c=self.c, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'genpareto'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v])
        try:
            scale_ = self.scale
        except:
            scale_ = 1
        out = (scale_ / (self.c - 1)) * ((1 + self.c * v / scale_) ** (1 - 1 / self.c) - 1)
        out[v < 0] = v[v < 0]
        return out

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            scale_ = self.scale
        except:
            scale_ = 1
        self.scale = scale_
        loc = (loc - self.loc)
        # return 1 - scipy.stats.genpareto(loc=0, c=self.c, scale=self.scale).cdf(d - loc)
        return 1 - self.cdf(low - loc)

## Lognormal
class Lognormal(_ContinuousDistribution):
    """
    Lognormal distribution.
    scipy reference distribution: ``scipy.stats._continuous_distns.lognorm_gen ``

    :param scale: lognormal scale parameter.
    :type scale: ``float``
    :param loc: lognormal location parameter.
    :type loc:``float``
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *s* (``int`` or ``float``) --
          shape parameter s.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        super().__init__()
        self.s = kwargs['s']
        self.scale = scale
        self.loc = loc

    @property
    def s(self):
        return self.__s

    @s.setter
    def s(self, value):
        assert isinstance(value, (float, int)), logger.error("s has to be float or int type")
        self.__s = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)), logger.error("scale has to be float or int type")
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error("loc has to be float or int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.lognorm(s=self.s, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'lognormal'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v])
        out = v.copy()
        try:
            loc = np.log(self.scale)
        except:
            loc = 0
        shape = self.s
        out[v > 0] = np.exp(loc + shape ** 2 / 2) * (scipy.stats.norm.cdf((np.log(v[v > 0]) - (loc + shape ** 2)) / shape)) + \
                     v[v > 0] * (
                             1 - scipy.stats.norm.cdf((np.log(v[v > 0]) - loc) / shape))
        return out

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            loc_ = np.log(self.scale)
        except:
            loc_ = 0
        self.scale = np.exp(loc_)

        loc = (loc - self.loc)

        # return 1 - scipy.stats.lognorm(scale=np.exp(loc_), s=self.s).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)

## Generalized beta
class GenBeta():
    """
    Generalized Beta (GB) distribution, also refer to as Generalized Beta
    of the second kind, or the Generalized Beta Prime distribution. 
    If X is a GB distributed r.v., its cumulative distribution function can
    be expressed as:
    
    Pr[X <= x] = Pr[Y <= (x/scale)^shape3], 0 < x < scale,

    where Y has a Beta distribution, with parameters shape1 and shape2.
    Refer to Appendix A of Klugman, Panjer & Willmot, Loss Models, Wiley.
    """

    def __init__(self, shape1, shape2, shape3, scale):
        
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        self.scale = scale

    @property
    def shape1(self):
        return self.__shape1

    @shape1.setter
    def shape1(self, value):
        assert isinstance(value, (float, int)), logger.error("shape1 has to be float or int")
        assert (value > 0), logger.error("shape1 has to be > 0")
        self.__shape1 = value

    @property
    def shape2(self):
        return self.__shape2

    @shape2.setter
    def shape2(self, value):
        assert isinstance(value, (float, int)), logger.error("shape2 has to be float or int")
        assert (value > 0), logger.error("shape2 has to be > 0")
        self.__shape2 = value

    @property
    def shape3(self):
        return self.__shape3

    @shape3.setter
    def shape3(self, value):
        assert isinstance(value, (float, int)), logger.error("shape3 has to be float or int")
        assert (value > 0), logger.error("shape3 has to be > 0")
        self.__shape3 = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)), logger.error("scale has to be float or int")
        assert (value > 0), logger.error("scale has to be > 0")
        self.__scale = value

    @property
    def dist(self):
        return scipy.stats.beta(self.shape1, self.shape2)

    @staticmethod
    def category():
        return {'severity'}

    @staticmethod
    def name():
        return 'genbeta'

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')

        random_state = int(time.time()) if (random_state is None) else random_state
        np.random.seed(random_state)

        tmp_ = scipy.stats.beta(a=self.shape1, b=self.shape2).rvs(size=size, random_state=random_state)
        return self.scale * pow(tmp_, 1.0/self.shape3)
        
    def pdf(self, x):
        """
        Probability density function.

        :param x: quantile where probability density function is evaluated.
        :type x: ``float``
        :return: probability density function in x.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        x = hf.arg_type_handler(x)
        x_shape = len(x) 
        output = np.zeros(x_shape)

        # if (x < 0.0) or (x > self.scale) return 0 # not needed, it is already zero
        
        filter_one = (x == 0.0)
        if np.any(filter_one):
            # output[filter_x_is_zero * (self.shape1 * self.shape3 > 1)] = 0 # not needed, it is already zero
            output[filter_one * (self.shape1 * self.shape3 < 1)] = np.infty
            output[filter_one * (self.shape1 * self.shape3 == 1)] = self.shape3 / scipy.special.beta(self.shape1, self.shape2)
        
        filter_two = (x > 0.0) * (x < self.scale)
        if np.any(filter_two):
            x_ = x[filter_two]
            logu = self.shape3 * (np.log(x_) - np.log(self.scale))
            log1mu = np.log1p(-np.exp(logu))
            tmp = np.exp(np.log(self.shape3) + self.shape1 * logu + (self.shape2 - 1.0) * \
                log1mu - np.log(x_) - scipy.special.betaln(self.shape1, self.shape2))
            output[filter_two] = tmp

        filter_three = (x > 0.0) * (x == self.scale)
        if np.any(filter_three):
            # output[filter_x_is_selfscale * (self.shape2 > 1)] = 0 # not needed, it is already zero
            output[filter_three * (self.shape2 < 1)] = np.infty
            output[filter_three * (self.shape2 == 1)] = self.shape1 * self.shape3
        
        if len(output) == 1:
            output = output.item()

        return output

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in x.
        :type x: ``float``
        :return: cumulative distribution function in x.
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        x = hf.arg_type_handler(x)
        x_shape = len(x) 
        output = np.zeros(x_shape)

        # if (x <= 0.0) return 0 # not needed, it is already zero

        filter_one = (x > 0.0) * (x < self.scale)
        if np.any(filter_one):
            u = np.exp(self.shape3 * (np.log(x[filter_one]) - np.log(self.scale)))
            output[filter_one] = self.dist.cdf(u)
        
        filter_two = (x >= self.scale)
        if np.any(filter_two):
            output[filter_two] = 1

        if len(output) == 1:
            output = output.item()
        return output

    def logpdf(self, x):
        """
        Natural logarithm of the probability distribution function.

        :param x: natural logarithm of the probability distribution function computed in x.
        :type x: ``float``
        :return: natural logarithm of the probability distribution function computed in x.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return np.log(self.pdf(x))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: natural logarithm of the cumulative distribution function computed in x.
        :type x: ``float``
        :return: natural logarithm of the cumulative distribution function in x.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def sf(self, x):
        """
        Survival function, 1 - cumulative distribution function.

        :param x: survival function will be computed in x.
        :type x: ``float``
        :return: survival function in x
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return 1 - self.cdf(x)

    def logsf(self, x):
        """
        Natural logarithm of the survival function.

        :param x: natural logarithm of the survival function computed in x.
        :type x: ``float``
        :return: natural logarithm of the survival function in x
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.sf(x=x))

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        return self.scale * pow(self.dist.ppf(q=q), 1.0/self.shape3)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.ppf(1-q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        if (n < self.shape1 * self.shape3) and (n < -self.shape2 * self.shape3):
            return np.inf
        tmp_ = n / self.shape3

        return pow(self.scale, n) * scipy.special.beta(self.shape1 + tmp_, self.shape2) / \
            scipy.special.beta(self.shape1, self.shape2)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        t_ = []
        if 'm' in moments:
            t_.append(self.mean())
        if 'v' in moments:
            t_.append(self.var())
        if 's' in moments:
            t_.append(self.moment(3) / self.moment(2)**(3/2))
        if 'k' in moments:
            t_.append(self.moment(4) / self.moment(2)**2 - 3)
        assert len(t_) > 0, "moments argument is not composed of letters 'mvsk'"

        return tuple(t_)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.ppf(p=0.5)

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.moment(1)

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.moment(2) - self.moment(1)**2

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.var()**(1/2)

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """

        v = hf.arg_type_handler(v)
        v_shape = len(v) 
        output = np.zeros(v_shape)
        
        filter_ = (v > 0.0)
        if np.any(filter_):
            v_ = v[filter_]
            # z_ = 0 if np.isinf(v_) else v_
            z_ = v_.copy()
            z_[np.isinf(v_)] = 0
            tmp_ = 1 / self.shape3
            u_ = np.exp(self.shape3 * (np.log(v_) - np.log(self.scale)))
            
            output[filter_] = self.scale * scipy.special.beta(self.shape1 + tmp_, self.shape2) / \
                scipy.special.beta(self.shape1, self.shape2) * scipy.stats.beta.cdf(u_, self.shape1 + tmp_, self.shape2) \
                    + z_ * (1-self.dist.cdf(u_))
        
        if (1 <= - self.shape1 * self.shape3):
            output = [np.infty] * v_shape

        if len(output) == 1:
            output = output.item()

        return output 

    def den(self, low):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            scale_ = self.scale
        except:
            scale_ = 1
        self.scale = scale_
        
        return 1 - self.cdf(low)

## Burr
class Burr12(_ContinuousDistribution):
    """
    Burr distribution, also referred to as the Burr Type XII, Singh–Maddala distribution.
    When d=1, this is a Fisk distribution.
    When c=d, this is a Paralogistic distribution.
    scipy reference distribution: ``scipy.stats._continuous_distns.burr_gen object``

    :param scale: burr scale parameter.
    :type scale: ``float``
    :param loc: burr location parameter.
    :type loc: ``float``
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter c.
        * *d* (``int`` or ``float``) --
          shape parameter d.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        super().__init__()
        self.c = kwargs['c']
        self.d = kwargs['d']
        self.scale = scale
        self.loc = loc

    @property
    def c(self):
        return self.__c
    
    @c.setter
    def c(self, value):
        assert isinstance(value, (float, int)), logger.error("c has to be float or int type")
        self.__c = value

    @property
    def d(self):
        return self.__d

    @c.setter
    def d(self, value):
        assert isinstance(value, (float, int)), logger.error("d has to be float or int type")
        self.__d = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error("loc has to be float or int type")
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)), logger.error("scale has to be float or int type")
        self.__scale = value

    @property
    def dist(self):
        return scipy.stats.burr12(c=self.c, d=self.d, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'burr12'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v]).flatten()
        out = v.copy()
        u = v.copy()
        u[v>0] = 1 / (1 + (v[v>0] / self.scale) ** self.c)
        out[v>0] = v[v>0]*(u[v>0]**self.d)+scipy.special.betaincinv(1+1/self.c,self.d-1/self.c,1-u[v>0])*(self.scale*scipy.special.gamma(1+1/self.c)*scipy.special.gamma(self.d-1/self.c)/scipy.special.gamma(self.d))
        return out


    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        # change both d and loc names
        loc = (loc - self.loc)
        # return 1 - scipy.stats.burr12(c=self.c,d=self.d,scale=self.scale).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)

## Dagum
class Dagum(_ContinuousDistribution):
    """
    Wrapper to scipy mielke distribution.
    It is referred to the Inverse Burr, Mielke Beta-Kappa.
    When d=s, this is an inverse paralogistic.
    ``scipy.stats._continuous_distns.lognorm_gen ``

    :param scale: dagum scale parameter.
    :type scale: ``float``
    :param loc: dagum location parameter.
    :type loc: ``float``
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *d* (``int`` or ``float``) --
          shape parameter d.
        * *s* (``int`` or ``float``) --
          shape parameter s.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        super().__init__()
        self.d = kwargs['d']
        self.s = kwargs['s']
        self.scale = scale
        self.loc = loc

    @property
    def d(self):
        return self.__d

    @d.setter
    def d(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'd must be a positive number')
        self.__d = value

    @property
    def s(self):
        return self.__s

    @s.setter
    def s(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            's must be a positive number')
        self.__s = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'scale must be a positive number')
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error("loc has to be float or int type")
        self.__loc = value

    @property
    def k(self):
        return self.d * self.s

    @property
    def dist(self):
        return scipy.stats.mielke(
            k=self.k,
            s=self.s,
            loc=self.loc,
            scale=self.scale
            )

    @staticmethod
    def name():
        return 'dagum'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v]).flatten()
        output = v.copy()
        u = v.copy()
        u[v > 0] = (v[v > 0] / self.scale) ** self.s / (1 + (v[v > 0] / self.scale) ** self.s)
        output[v > 0] = v[v > 0]*(1-u[v > 0]**self.d) +\
            scipy.special.betaincinv(self.d + 1/self.s, 1 - 1/self.s, u[v > 0])*\
                (self.scale*scipy.special.gamma(self.d + 1/self.s)*\
                    scipy.special.gamma(1 - 1/self.s)/scipy.special.gamma(self.d))
        return output

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        loc = (loc - self.loc)
        # return 1 - scipy.stats.mielke(k=self.x, s=self.s, scale=self.scale).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)

## Weibull
class Weibull(_ContinuousDistribution):
    """
    Wrapper to scipy Weibull (Weibull_min) distribution.
    ``scipy.stats._continuous_distns.weibull_min_gen object``

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter c.

    """

    def __init__(self, loc=0, scale=1, **kwargs):
        super().__init__()
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'c must be a positive number')
        self.__c = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'scale must be a positive number')
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error("loc has to be float or int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.weibull_min(
            c=self.c,
            loc=self.loc,
            scale=self.scale
            )

    @staticmethod
    def name():
        return 'weibull'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v]).flatten()
        output = v.copy()
        output[v > 0] = v[v > 0] * np.exp(-(v[v > 0]/self.scale)**self.c)+\
            self.scale*scipy.special.gamma(1 + 1/self.c)*\
                scipy.special.gammainc(1 + 1/self.c, (v[v > 0]/self.scale)**self.c)
        return output

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        loc = (loc - self.loc)
        # return 1 - scipy.stats.weibull_min(c=self.c, scale=self.scale).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)

## Inverse Weibull
class InvWeibull(_ContinuousDistribution):
    """
    Wrapper to scipy inverse Weibull distribution.
    ``scipy.stats._continuous_distns.invweibull_gen object``

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter c.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        super().__init__()
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'c must be a positive number')
        self.__c = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'scale must be a positive number')
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error("loc has to be float or int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.invweibull(
            c=self.c,
            loc=self.loc,
            scale=self.scale
            )

    @staticmethod
    def name():
        return 'invweibull'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v]).flatten()
        output = v.copy()
        output[v > 0]=v[v > 0]*(1-np.exp(-(self.scale/v[v>0])**self.c))+\
            self.scale * scipy.special.gamma(1-1/self.c)*\
                (1 - scipy.special.gammainc(1-1/self.c,(self.scale/v[v>0])**self.c))
        return output

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        loc = (loc - self.loc)
        # return 1 - stats.invweibull(c=self.c,scale=self.scale).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)

## Inverse Gamma
class InvGamma(_ContinuousDistribution):
    """
    Wrapper to scipy inverse gamma distribution.
    ``scipy.stats._continuous_distns.invgamma_gen object``
    
    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
        :param \**kwargs:
    See below

    :Keyword Arguments:
        * *a* (``int`` or ``float``) --
          shape parameter a.
    """

    def __init__(self,loc=0,scale=1,**kwargs):
        super.__init__()
        self.a=kwargs['a']
        self.scale = scale
        self.loc = loc

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'a must be a positive number')
        self.__a = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error(
            'scale must be a positive number')
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error("loc has to be float or int type")
        self.__loc = value

    @property
    def dist(self):
        return scipy.stats.invgamma(
            a=self.a,
            loc=self.loc,
            scale=self.scale
            )

    @staticmethod
    def name():
        return 'invgamma'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        v = np.array([v]).flatten()
        output = v.copy()
        output[v > 0] = v[v > 0]*scipy.special.gammainc(self.a, self.scale/v[v > 0])+\
            self.scale*(1-scipy.special.gammainc(self.a-1, self.scale/v[v > 0]))*\
                scipy.special.gamma(self.a-1)/scipy.special.gamma(self.a)
        return output

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        loc = (loc - self.loc)
        #  return 1 - stats.invgamma(loc=0,a=self.a,scale= self.scale).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)

## Inverse Gaussian
class InvGauss(_ContinuousDistribution):
    """
    Wrapper to scipy inverse gaussian distribution.
    ``scipy.stats._continuous_distns.invgauss_gen object``
    
    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
        :param \**kwargs:
    See below

    :Keyword Arguments:
        * *mu* (``int`` or ``float``) --
          shape parameter mu.
    """

    def __init__(self, loc=0., scale=1., **kwargs):
        super().__init__()
        self.scale = scale
        self.loc = loc
        self.mu = kwargs['mu']

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        assert isinstance(value, (int, float)), logger.error("mu has to be float or int type")
        self.__mu = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (int, float)), logger.error("loc has to be float or int type")
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (int, float)), logger.error("scale has to be float or int type")
        self.__scale = value

    @property
    def dist(self):
        return scipy.stats.invgauss(mu=self.mu, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'invgauss'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        try:
            scale_ = self.scale
        except:
            scale_ = 1

        v = np.array([v]).flatten()
        output = v.copy()
        z = v.copy()
        y = v.copy()
        z[v > 0] = (v[v > 0] - self.mu) / self.mu
        y[v > 0] = (v[v > 0] + self.mu) / self.mu
        output[v > 0]= v[v > 0] - self.mu * z[v > 0] *\
            scipy.stats.norm.cdf(z[v > 0] *\
                np.sqrt(1 / v[v > 0])) - self.mu * y[v > 0] * np.exp(2 / self.mu) *\
                    scipy.stats.norm.cdf(-y[v > 0] * np.sqrt(1 / v[v > 0]))
        return scale_ * output

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            scale_ = self.scale
        except:
            scale_ = 1

        self.scale = scale_ 
        loc = (loc - self.loc)
        # return 1 - scipy.stats.invgauss(loc=0, mu=self.mu, scale=scale_).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)

## Fisk 
class Fisk(_ContinuousDistribution):
    """
    Wrapper to scipy Fisk distribution.
    ``scipy.stats._continuous_distns.fisk_gen object``

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
        :param \**kwargs:
    See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter c.
    """

    def __init__(self, loc=0., scale=1., **kwargs):
        super().__init__()
        self.scale = scale
        self.loc = loc
        self.c = kwargs['c']

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, (float, int)), logger.error("loc has to be float or int type")
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (float, int)), logger.error("scale has to be float or int type")
        self.__scale = value

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, value):
        assert isinstance(value, (float, int)) and value >= 0, logger.error('c must be a positive number')
        self.__c = value

    @property
    def dist(self):
        return scipy.stats.fisk(c=self.c, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'fisk'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        try:
            scale_ = self.scale
        except:
            scale_ = 1

        v = np.array([v]).flatten()
        output = v.copy()
        u = v.copy()
        u[v > 0] = (v[v > 0]**self.c) / (1+v[v > 0]**self.c)
        
        output[v > 0] = v[v > 0]*(1-u[v > 0]) +\
            scale_ * scipy.special.gamma(1 + 1 / self.c) *\
                scipy.special.gamma(1 - 1 / self.c) *\
                    scipy.special.betaincinv(1 + 1/self.c, 1 - 1/self.c, u[v > 0])
        return output

    def den(self, low, loc):
        """
        It returns the denominator of the local moments discretization.

        :param low: lower priority.
        :type low: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            scale_ = self.scale
        except:
            scale_ = 1
        self.scale = scale_
        loc = (loc - self.loc)
        # return 1 - scipy.stats.fisk(loc=0, c=self.c, scale=scale_).cdf(d - loc)
        return 1 - self.dist.cdf(low - loc)
