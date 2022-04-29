import numpy as np
import scipy.stats
from scipy import stats,special
from twiggy import quick_setup,log

quick_setup()
logger= log.name('distributions')

# Wrappers to scipy
## poisson distribution

class Poisson():
    """
    Wrapper to scipy poisson distribution.

    :param mu: poisson distribution parameter mu.
    :type mu: ``float``
    :param loc: location parameter of the poisson distribution.
    :type loc: ``float``
    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param p0: RV probability in zero.
    :type p0: ``float``

    """
    name='poisson'
    def __init__(self,mu,loc=0):
        self.mu=mu
        self.loc=loc
        self.__dist=scipy.stats.poisson(mu=self.mu,loc=self.loc)
        self.a=0
        self.b= self.mu
        self.p0=np.array(np.exp(-self.mu))

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: int
        :param random_state: random state for the random number generator.
        :type random_state: int

        :return: Random variates.
        :rtype: int or numpy.ndrarry

        """
        return self.__dist.rvs(size=size,random_state=random_state)

    def pmf(self,k):
        """
        Probability mass function

        :param k: probability mass function will be computed in k.
        :type k: int

        :return: pmf
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.__dist.pmf(k=k)

    def logpmf(self,k):
        """
        Log of the probability mass function.

        :param k: log of the probability mass function computed in k.
        :type k: int
        :return: log pmf
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.__dist.logpmf(k=k)

    def cdf(self,k):
        """
        Cumulative distribution function.

        :param k: cumulative density function will be computed in k.
        :type k: int
        :return: cdf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.cdf(k)

    def logcdf(self,k):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in k.
        :type k: int
        :return: cdf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.logcdf(k)

    def sf(self,k):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param k: survival function will be computed in k.
        :type k: int
        :return: sf
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.__dist.sf(k)

    def logsf(self, k):
        """
        Log of the survival function.

        :param k:log of the survival function computed in k.
        :type k: int
        :return: cdf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.logsf(k)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.isf(q=q)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: numpy.ndarray
        """
        return self.__dist.entropy()

    def expect(self, func, args=None, lb=None, ub=None, conditional=False):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.mu,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: numpy.float64
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: numpy.float64
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: numpy.float64
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: numpy.float64
        """
        return self.__dist.std()

    def interval(self,alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def pgf(self,f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: numpy.ndarray

        """
        return np.exp(self.b * (f - 1))

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: int
        """
        return self.a,self.b,self.p0

## binomial distribution

class Binom():
    """
    Wrapper to scipy binomial distribution.

    :param n: number of trials.
    :type n: ``int``
    :param p: probability parameter of the binomial distribution.
    :type p: ``float``
    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param p0: RV probability in zero.
    :type p0: ``float``

    """
    name='binom'
    def __init__(self,n,p,loc=0):
        self.n = n
        self.p = p
        self.loc=loc
        self.__dist=scipy.stats.binom(n=self.n,p=self.p,loc=self.loc)
        self.a= -self.p / (1 - self.p)
        self.b= (self.n + 1) * (self.p / (1 - self.p))
        self.p0= (1 - self.p) ** self.n


    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: int
        :param random_state: random state for the random number generator.
        :type random_state: int

        :return: Random variates.
        :rtype: int or numpy.ndrarry

        """
        return self.__dist.rvs(size=size,random_state=random_state)

    def pmf(self,k):
        """
        Probability mass function.

        :param k: probability mass function computed in k.
        :type k: int

        :return: probability mass function
        :rtype: numpy.ndarray
        """
        return self.__dist.pmf(k=k)

    def logpmf(self,k):
        """
        Log of the probability mass function.

        :param k: log of the probability mass function computed in k.
        :type k: int
        :return: log of the probability mass function
        :rtype: numpy.ndarray
        """
        return self.__dist.logpmf(k=k)

    def cdf(self,k):
        """
        Cumulative distribution function.

        :param k: cumulative density function computed in k.
        :type k: int
        :return: cumulative density function
        :rtype: numpy.ndarray
        """
        return self.__dist.cdf(k)

    def logcdf(self,k):
        """
        Log of the cumulative distribution function.

        :param k: The logarithm of the cumulative distribution computed in k.
        :return: cdf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.logcdf(k)

    def sf(self,k):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param k: The survival function will be computed in k.
        :return: sf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.sf(k)

    def logsf(self, k):
        """
        Log of the survival function.

        :param k: The logarithm of the survival function computed in k.
        :return:logsf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.logsf(k)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: The percent point function will be computed in q.
        :return: ppf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.isf(q=q)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: numpy.ndarray
        """
        return self.__dist.entropy()

    def expect(self, func, args=None, lb=None, ub=None, conditional=False):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.
        """
        if args is None:
            args = (self.mu,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: numpy.float64
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: numpy.float64
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: numpy.float64
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: numpy.float64
        """
        return self.__dist.std()

    def interval(self,alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def pgf(self,f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: numpy.ndarray

        """
        return (1+self.a/(self.a-1)*(f-1))**(-self.b/self.a-1)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: int
        """
        return self.a,self.b,self.p0

## geometric

class Geom():
    """
    Wrapper to scipy geometric distribution.

    :param p: probability parameter of the geometric distribution.
    :type p: ``float``
    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param p0: RV probability in zero.
    :type p0: ``float``

    """
    name='geom'

    def __init__(self,p,loc=0):
        self.p = p
        self.loc=loc
        self.__dist=scipy.stats.geom(p=self.p,loc=self.loc)
        self.a= 1-self.p
        self.b= 0
        self.p0= np.array([((1-self.p)/self.p)**-1])


    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: int
        :param random_state: random state for the random number generator.
        :type random_state: int

        :return: Random variates.
        :rtype: int or numpy.ndrarry

        """
        return self.__dist.rvs(size=size,random_state=random_state)

    def pmf(self,k):
        """
        Probability mass function.

        :param k: probability mass function computed in k.
        :type k: int

        :return: probability mass function
        :rtype: numpy.ndarray
        """
        return self.__dist.pmf(k=k)

    def logpmf(self,k):
        """
        Log of the probability mass function.

        :param k: log of the probability mass function computed in k.
        :type k: int
        :return: log of the probability mass function
        :rtype: numpy.ndarray
        """
        return self.__dist.logpmf(k=k)

    def cdf(self,k):
        """
        Cumulative distribution function.

        :param k: cumulative density function computed in k.
        :type k: int
        :return: cumulative density function
        :rtype: numpy.ndarray
        """
        return self.__dist.cdf(k)

    def logcdf(self,k):
        """
        Log of the cumulative distribution function.

        :param k: The logarithm of the cumulative distribution computed in k.
        :return: cdf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.logcdf(k)

    def sf(self,k):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param k: The survival function will be computed in k.
        :return: sf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.sf(k)

    def logsf(self, k):
        """
        Log of the survival function.

        :param k: The logarithm of the survival function computed in k.
        :return:logsf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.logsf(k)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: The percent point function will be computed in q.
        :return: ppf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.isf(q=q)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: numpy.ndarray
        """
        return self.__dist.entropy()

    def expect(self, func, args=None, lb=None, ub=None, conditional=False):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.
        """
        if args is None:
            args = (self.mu,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: numpy.float64
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: numpy.float64
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: numpy.float64
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: numpy.float64
        """
        return self.__dist.std()

    def interval(self,alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def pgf(self,f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: numpy.ndarray

        """
        return (1-self.a/(1-self.a)*(f-1))**(-1)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: int
        """
        return self.a,self.b,self.p0

## negative binomial

class Nbinom():
    """
    Wrapper to scipy negative binomial distribution.

    :param n: size parameter of the negative binomial distribution.
    :type n: ``int``
    :param p: probability parameter of the negative binomial distribution.
    :type p: ``float``
    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param p0: RV probability in zero.
    :type p0: ``float``

    """
    name = 'nbinom'

    def __init__(self,n, p, loc=0):
        self.n= n
        self.p = p
        self.loc = loc
        self.__dist = scipy.stats.nbinom(n=self.n,p=self.p, loc=self.loc)
        self.a = 1-self.p
        self.b = (self.n-1)*(1-self.p)
        self.p0 = np.array([self.p**self.n])

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: int
        :param random_state: random state for the random number generator.
        :type random_state: int

        :return: Random variates.
        :rtype: int or numpy.ndrarry

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pmf(self, k):
        """
        Probability mass function.

        :param k: probability mass function computed in k.
        :type k: int

        :return: probability mass function
        :rtype: numpy.ndarray
        """
        return self.__dist.pmf(k=k)

    def logpmf(self, k):
        """
        Log of the probability mass function.

        :param k: log of the probability mass function computed in k.
        :type k: int
        :return: log of the probability mass function
        :rtype: numpy.ndarray
        """
        return self.__dist.logpmf(k=k)

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative density function computed in k.
        :type k: int
        :return: cumulative density function
        :rtype: numpy.ndarray
        """
        return self.__dist.cdf(k)

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: The logarithm of the cumulative distribution computed in k.
        :return: cdf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.logcdf(k)

    def sf(self, k):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param k: The survival function will be computed in k.
        :return: sf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.sf(k)

    def logsf(self, k):
        """
        Log of the survival function.

        :param k: The logarithm of the survival function computed in k.
        :return:logsf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.logsf(k)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: The percent point function will be computed in q.
        :return: ppf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: numpy.float64 or numpy.ndarray
        """
        return self.__dist.isf(q=q)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: numpy.ndarray
        """
        return self.__dist.entropy()

    def expect(self, func, args=None, lb=None, ub=None, conditional=False):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.
        """
        if args is None:
            args = (self.mu,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: numpy.float64
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: numpy.float64
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: numpy.float64
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: numpy.float64
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def pgf(self, f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: numpy.ndarray

        """
        return (1-self.a/(1-self.a)*(f-1))**(-self.b/self.a-1)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: int
        """
        return self.a, self.b, self.p0

#### ZT and ZM classes

class ZTpoisson:
    """
    Zero-Truncated Poisson distribution.

    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param __p0: base RV probability in zero.
    :type __p0: ``float``
    :param __temp: base RV distribution.
    :type __temp: ``float``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          ZT poisson distribution parameter mu.
    """
    name = 'ZTpoisson'

    def __init__(self, loc=0, **kwargs):
        self.mu = kwargs['mu']
        self.a=0
        self.b=self.mu
        self.__p0 = np.exp(-self.mu)
        self.__temp = stats.poisson(mu=self.mu, loc=loc)

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, var):
        assert var > 0, logger.error('The Zero-Truncated Poisson parameter must be positive')
        self.__mu = var

    def pmf(self, k):
        """
        Probability mass function.

        :param k: probability mass function will be computed in k.
        :type k: ``int`` or ``numpy.ndarray``
        :return: pmf.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        temp = (self.__temp.pmf(k)) / (1 - self.__p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative distribution function will be computed in k.
        :type k: ``int`` or ``numpy.ndarray``
        :return: cdf.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self.__temp.cdf(k) - self.__temp.cdf(0)) / (1 - self.__temp.cdf(0))

    def rvs(self, size,random_state=None):
        """
        Random variates.

        :param size: random draws sample size.
        :type size: ``int``
        :return: random draws.
        :rtype: ``numpy.ndarray``
        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')
        np.random.seed(random_state)
        q_=np.random.uniform(low=self.__temp.cdf(0),high=1,size=size)
        return self.__temp.ppf(q_)
        # n=int(n)
        # p0=self.__temp.pmf(0) # ricordati di metterlo attributo: se eseguo self.rvs() 1000 volte lo definisci 1000 volte?
        # nsim = int(n+np.ceil(1.10*p0*n)) # non rigoroso
        # r_=self.__temp.rvs(nsim)
        # r_=r_[r_ > 0]
        # while r_.shape[0] < n:
        #     l_= int(np.maximum(np.ceil(1.10*(n-r_.shape[0])*p0),10)) #non rigoroso
        #     r_=np.concatenate((r_,self.__temp.rvs(l_)))
        #     r_=r_[r_>0]
        # r_=r_[:n]
        # return (r_)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param p_: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        p_=np.array(q)
        return (self.__temp.ppf(q=p_ * (1 - self.__temp.cdf(0)) + self.__temp.cdf(0)))

    def pgf(self,f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: numpy.ndarray

        """
        return (np.exp(self.b * f) - 1) / (np.exp(self.b) - 1)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: int
        """
        return self.a,self.b,0

####################

class ZMpoisson:
    """
    Zero-modified Poisson distribution.

    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param __p0: base RV probability in zero.
    :type __p0: ``float``
    :param __temp: base RV distribution.
    :type __temp: ``scipy.stats._discrete_distns.poisson_gen``
    :param __ZTtemp: correspondent ZT distribution.

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          ZM poisson distribution parameter mu.
        * *p0M* (``numpy.float64``) --
          ZM poisson mixing parameter.
    """
    name = 'ZMpoisson'
    def __init__(self,loc=0, **kwargs):
        self.mu = kwargs['mu']
        self.loc=loc
        self.p0M = kwargs['p0M']
        self.a=0
        self.b=self.mu
        self.__p0 = np.exp(-self.mu)
        self.__temp = stats.poisson(mu=self.mu,loc=self.loc)
        self.__tempZT= ZTpoisson(mu=self.mu,loc=self.loc)

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, var):
        assert var > 0, logger.error('The Zero-Modified Poisson parameter must be positive')
        self.__mu = var

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Modified Poisson parameter p0M must be between zero and one.')
        self.__p0M = var

    def pmf(self, k):
        """
        Probability mass function

        :param k: probability mass function will be computed in k.
        :type k: int

        :return: pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        # return (self.__temp.pmf(x)*(1-self.p0M))/(1-self.__p0)
        temp = (self.__temp.pmf(k) * (1 - self.p0M)) / (1 - self.__p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = self.p0M
            else:
                temp[zeros] = self.p0M
            return temp

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative density function will be computed in k.
        :type k: ``int`` or ``numpy.ndarray``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.p0M + (1 - self.p0M) * (self.__temp.cdf(k) - self.__temp.cdf(0)) / (1 - self.__temp.cdf(0))

    # def __ZTrvs(self, n):
    #     n=int(n)
    #     distTemp=stats.poisson(mu=self.mu)
    #     p0=distTemp.pmf(0)
    #     nsim = int(n+np.ceil(1.10*p0*n)) #non rigoroso
    #     r_=distTemp.rvs(nsim)
    #     r_=r_[r_ > 0]
    #     while r_.shape[0] < n:
    #         l_= int(np.maximum(np.ceil(1.10*(n-r_.shape[0])*p0),10)) #non rigoroso
    #         r_=np.concatenate((r_,distTemp.rvs(l_)))
    #         r_=r_[r_>0]
    #     r_=r_[:n]
    #     return (r_)

    def rvs(self, size,random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``int`` or ``numpy.ndrarry``
        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')
        r_= stats.bernoulli(p=1-self.p0M).rvs(int(size),random_state=random_state)
        c_= np.where(r_==1)[0]
        if int(len(c_)) > 0:
            r_[c_]= self.__tempZT.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """

        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :type q:``numpy.float64`` or ``numpy.ndarray``
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        p_ = np.array(q)
        temp = self.__temp.ppf((1 - self.__temp.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.__temp.cdf(0))

        # zeros = np.where(p_ <= self.p0M)[0]
        # if zeros.size == 0:
        #     return temp
        # else:
        #     if p_.shape == ():
        #         temp = self.p0M
        #     else:
        #         temp[zeros] = self.p0M
        #     return temp
        return temp

    def pgf(self,f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: ``numpy.ndarray``

        """
        return self.p0M+(1-self.p0M)*((np.exp(self.b*f)-1)/(np.exp(self.b)-1))

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a,self.b,self.p0M #+ (1-self.p0M)*self.__temp.pmf(0)

#############

class ZTbinom:
    """
    Zero-truncated binomial distribution.

    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param __p0: base RV probability in zero.
    :type __p0: ``float``
    :param __temp: base RV distribution.
    :type __temp: ``float``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``numpy.float64``) --
          ZT binomial distribution size parameter n.
        * *p*(``numpy.float64``) --
           ZT binomial distribution probability parameter p.
    """
    name = 'ZTbinom'
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.a= -self.p / (1 - self.p)
        self.b= (self.n + 1) * (self.p / (1 - self.p))
        self.__p0 = np.array([(1 - self.p) ** self.n])
        self.__temp = stats.binom(n=self.n, p=self.p)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, var):
        assert isinstance(var, int) and var >= 1, logger.error(
            'The Zero-Truncated Binomial parameter n must be a natural number')
        self.__n = var

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Truncated Binomial parameter p must be between zero and one.')
        self.__p = var

    def pmf(self, k):
        """
        Probability mass function

        :param k: probability mass function will be computed in k.
        :type k: int

        :return: pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        # temp = (stats.binom(n=self.n, p=self.p).pmf(x)) / (1 - self.__p0)
        temp = (self.__temp.pmf(k)) / (1 - self.__p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative density function will be computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self.__temp.cdf(k) - self.__temp.cdf(0)) / (1 - self.__temp.cdf(0))

    def rvs(self, size,random_state=None):
        """
        Random variates.

        :param size: random draws sample size.
        :type size: ``int``
        :return: random draws.
        :rtype: ``numpy.ndarray``
        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')
        np.random.seed(random_state)
        q_=np.random.uniform(low=self.__temp.cdf(0),high=1,size=size)
        return self.__temp.ppf(q_)

        # n=int(n)
        # p0=self.__temp.pmf(0)
        # nsim = int(n+np.ceil(1.10*p0*n)) #non rigoroso
        # r_=self.__temp.rvs(nsim)
        # r_=r_[r_ > 0]
        # while r_.shape[0] < n:
        #     l_= int(np.maximum(np.ceil(1.10*(n-r_.shape[0])*p0),10)) #non rigoroso
        #     r_=np.concatenate((r_,self.__temp.rvs(l_)))
        #     r_=r_[r_>0]
        # r_=r_[:n]
        # return (r_)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :type q:``numpy.float64`` or ``numpy.ndarray``
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        p_=np.array(q)
        return (self.__temp.ppf(q=p_ * (1 - self.__temp.cdf(0)) + self.__temp.cdf(0)))

    def pgf(self, f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: ``numpy.ndarray``

        """
        return ((1+self.a/(self.a-1)*(f-1))**(-self.b/self.a-1)-(1-self.a)**(self.b/self.a+1))/(1-(1-self.a)**(self.b/self.a+1))

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, 0


###############

class ZMbinom:
    """
    Zero-modified binomial distribution.

    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param __p0: base RV probability in zero.
    :type __p0: ``float``
    :param __temp: base RV distribution.
    :type __temp: ``scipy.stats._discrete_distns.binom_gen``
    :param __ZTtemp: correspondent ZT distribution.

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``numpy.float64``) --
          ZM binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          ZM binomial distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          ZM binomial mixing parameter.
    """
    name = 'ZMbinom'
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.p0M = kwargs['p0M']
        self.a=-self.p / (1 - self.p)
        self.b=(self.n + 1) * (self.p / (1 - self.p))
        self.__p0 = np.array([(1 - self.p) ** self.n])
        # self.name = 'ZMbinom'
        self.__temp = stats.binom(n=self.n, p=self.p)
        self.__tempZT= ZTbinom(n=self.n, p=self.p)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, var):
        assert isinstance(var, int) and var >= 1, logger.error(
            'The Zero-Modified Binomial parameter n must be a natural number')
        self.__n = var

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Modified Binomial parameter p must be between zero and one.')
        self.__p = var

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Modified Binomial parameter p0M must be between zero and one.')
        self.__p0M = var

    def pmf(self, k):
        """
        Probability mass function

        :param k: probability mass function will be computed in k.
        :type k: int

        :return: pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return (self.__temp.pmf(k) * (1 - self.__p0M)) / (1 - self.__p0)

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative density function will be computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__p0M + (1 - self.__p0M) * (self.__temp.cdf(k) - self.__temp.cdf(0)) / (1 - self.__temp.cdf(0))

    # def __ZTrvs(self, n):
    #     n=int(n)
    #     distTemp=stats.binom(n=self.n,p=self.p)
    #     p0=distTemp.pmf(0)
    #     nsim = int(n+np.ceil(1.10*p0*n)) #non rigoroso
    #     r_=distTemp.rvs(nsim)
    #     r_=r_[r_ > 0]
    #     while r_.shape[0] < n:
    #         l_= int(np.maximum(np.ceil(1.10*(n-r_.shape[0])*p0),10)) #non rigoroso
    #         r_=np.concatenate((r_,distTemp.rvs(l_)))
    #         r_=r_[r_>0]
    #     r_=r_[:n]
    #     return (r_)

    def rvs(self, size,random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``int`` or ``numpy.ndrarry``

        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')

        r_= stats.bernoulli(p=1-self.p0M).rvs(size,random_state=random_state)
        c_= np.where(r_==1)[0]
        if int(len(c_)) > 0:
            r_[c_]= self.__tempZT.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :type q:``numpy.float64`` or ``numpy.ndarray``
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        p_ = np.array(q)
        temp = self.__temp.ppf((1 - self.__temp.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.__temp.cdf(0))
        return temp
        # zeros = np.where(p_ <= self.p0M)[0]
        # if zeros.size == 0:
        #     return temp
        # else:
        #     if p_.shape == ():
        #         temp = self.p0M
        #     else:
        #         temp[zeros] = self.p0M
        #     return temp


    def pgf(self,f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: ``numpy.ndarray``

        """
        return self.p0M+(1-self.p0M)*((1+self.a/(self.a-1)*(f-1))**(-self.b/self.a-1)-(1-self.a)**(self.b/self.a+1))/(1-(1-self.a)**(self.b/self.a+1))

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a,self.b,self.p0M#+(1-self.p0M)*self.__temp.pmf(0)

###################

class ZTgeom:
    """
    Zero-Truncated geometric distribution.

    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param __p0: base RV probability in zero.
    :type __p0: ``float``
    :param __temp: base RV distribution.
    :type __temp: ``float``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          ZT geometric distribution probability parameter p.
    """
    name = 'ZTgeom'
    def __init__(self, **kwargs):
        self.p = kwargs['p']
        self.a=1-self.p
        self.b=0
        self.__p0 = np.array([self.p])
        # self.name = 'ZTgeom'
        self.__temp = stats.geom(p=self.p)

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Truncated Geometric distribution parameter p must be between zero and one.')
        self.__p = var

    def pmf(self, k):
        """
        Probability mass function

        :param k: probability mass function will be computed in k.
        :type k: int

        :return: pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        temp = (self.__temp.pmf(k + 1)) / (1 - self.__p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative density function will be computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self.__temp.cdf(k) - self.__temp.cdf(0)) / (1 - self.__temp.cdf(0))

    def rvs(self, size,random_state=None):
        """
        Random variates.

        :param size: random draws sample size.
        :type size: ``int``
        :return: random draws.
        :rtype: ``numpy.ndarray``
        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')
        np.random.seed(random_state)
        q_ = np.random.uniform(low=self.__temp.cdf(0), high=1, size=size)
        return self.__temp.ppf(q_)
        # n=int(n)
        # p0=self.__temp.pmf(0)
        # nsim = int(n+np.ceil(1.10*p0*n)) #non rigoroso
        # r_=self.__temp.rvs(nsim)
        # r_=r_[r_ > 0]
        # while r_.shape[0] < n:
        #     l_= int(np.maximum(np.ceil(1.10*(n-r_.shape[0])*p0),10)) #non rigoroso
        #     r_=np.concatenate((r_,self.__temp.rvs(l_)))
        #     r_=r_[r_>0]
        # r_=r_[:n]
        # return (r_)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :type q:``numpy.float64`` or ``numpy.ndarray``
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        p_ = np.array(q)
        return (self.__temp.ppf(q=p_ * (1 - self.__temp.cdf(0)) + self.__temp.cdf(0)))

    def pgf(self, f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: ``numpy.ndarray``

        """
        return (1 / (1 - (f - 1) / (1 - self.a)) - 1 + self.a) / self.a

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, 0

########

class ZMgeom:
    """
    Zero-modified geometric distribution.

    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param __p0: base RV probability in zero.
    :type __p0: ``float``
    :param __temp: base RV distribution.
    :type __temp: ``scipy.stats._discrete_distns.geom_gen``
    :param __ZTtemp: correspondent ZT distribution.

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          ZM geometric distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          ZM geometric mixing parameter.
    """

    name = 'ZMgeom'
    def __init__(self, **kwargs):
        self.p = kwargs['p']
        self.p0M = kwargs['p0M']
        self.a=1-self.p
        self.b=0
        self.__p0 = np.array([self.p])
        self.__temp = stats.geom(p=self.p)
        self.__tempZT=ZTgeom(p=self.p)

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Modified Geometric distribution parameter p must be between zero and one.')
        self.__p = var

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Modified Geometric parameter p0M must be between zero and one.')
        self.__p0M = var

    def pmf(self, k):
        """
        Probability mass function

        :param k: probability mass function will be computed in k.
        :type k: int

        :return: pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return (self.__temp.pmf(k + 1) * (1 - self.__p0M)) / (1 - self.__p0)

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative density function will be computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__p0M + (1 - self.__p0M) * (self.__temp.cdf(k) - self.__temp.cdf(0)) / (1 - self.__temp.cdf(0))

    # def __ZTrvs(self, n):
    #     n=int(n)
    #     distTemp=stats.geom(p=self.p)
    #     p0=distTemp.pmf(0)
    #     nsim = int(n+np.ceil(1.10*p0*n)) #non rigoroso
    #     r_=distTemp.rvs(nsim)
    #     r_=r_[r_ > 0]
    #     while r_.shape[0] < n:
    #         l_= int(np.maximum(np.ceil(1.10*(n-r_.shape[0])*p0),10)) #non rigoroso
    #         r_=np.concatenate((r_,distTemp.rvs(l_)))
    #         r_=r_[r_>0]
    #     r_=r_[:n]
    #     return (r_)

    def rvs(self, size,random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``int`` or ``numpy.ndrarry``

        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')

        r_ = stats.bernoulli(p=1 - self.p0M).rvs(size,random_state=random_state)
        c_ = np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_] = self.__tempZT.rvs(int(len(c_)))
        return r_

    # def rvs(self, n):
    #     q_ = np.random.uniform(0, 1, n)
    #     return self.ppf(q_)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :type q:``numpy.float64`` or ``numpy.ndarray``
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        p_ = np.array(q)
        temp = self.__temp.ppf((1 - self.__temp.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.__temp.cdf(0))
        return temp
        # zeros = np.where(p_ <= self.p0M)[0]
        # if zeros.size == 0:
        #     return temp
        # else:
        #     if p_.shape == ():
        #         temp = self.p0M
        #     else:
        #         temp[zeros] = self.p0M
        #     return temp
        # return temp

    def pgf(self, f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: ``numpy.ndarray``

        """
        return self.p0M+(1-self.p0M)*(1/(1-(f-1)/(1-self.a))-1+self.a)/self.a

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, self.p0M#+(1-self.p0M)*self.__temp.pmf(0)

##############

class ZTnbinom:
    """
    Zero-Truncated Poisson distribution.

    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param __p0: base RV probability in zero.
    :type __p0: ``float``
    :param __temp: base RV distribution.
    :type __temp: ``float``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          ZT negative binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          ZT negative binomial distribution probability parameter p.
    """
    name = 'ZTnbinom'
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.a = 1 - self.p
        self.b = (self.n - 1) * (1 - self.p)
        self.__p0 = np.array([self.p ** self.n])
        # self.name = 'ZTnbinom'
        self.__temp = stats.nbinom(n=self.n, p=self.p)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, var):
        assert isinstance(var, int) and var >= 0, logger.error(
            'The Zero-Truncated Negative Binomial parameter n must be a positive number')
        self.__n = var

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Truncated Negative Binomial parameter p must be between zero and one.')
        self.__p = var

    def pmf(self, k):
        """
        Probability mass function

        :param k: probability mass function will be computed in k.
        :type k: int

        :return: pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        temp = (self.__temp.pmf(k)) / (1 - self.__p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative density function will be computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self.__temp.cdf(k) - self.__temp.cdf(0)) / (1 - self.__temp.cdf(0))

    def rvs(self, size,random_state=None):
        """
        Random variates.

        :param size: random draws sample size.
        :type size: ``int``
        :return: random draws.
        :rtype: ``numpy.ndarray``
        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')

        np.random.seed(random_state)
        q_ = np.random.uniform(low=self.__temp.cdf(0), high=1, size=size)
        return self.__temp.ppf(q_)
        # n=int(n)
        # p0=self.__temp.pmf(0)
        # nsim = int(n+np.ceil(1.10*p0*n)) #non rigoroso
        # r_=self.__temp.rvs(nsim)
        # r_=r_[r_ > 0]
        # while r_.shape[0] < n:
        #     l_= int(np.maximum(np.ceil(1.10*(n-r_.shape[0])*p0),10)) #non rigoroso
        #     r_=np.concatenate((r_,self.__temp.rvs(l_)))
        #     r_=r_[r_>0]
        # r_=r_[:n]
        # return (r_)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :type q:``numpy.float64`` or ``numpy.ndarray``
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        p_=np.array(q)
        return (self.__temp.ppf(q=p_ * (1 - self.__temp.cdf(0)) + self.__temp.cdf(0)))

    def pgf(self, f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: ``numpy.ndarray``

        """
        return ((1/(1-(f-1)*self.a/(1-self.a)))**(self.b/self.a+1)-(1-self.a)**(self.b/self.a+1))/(1-(1-self.a)**(self.b/self.a+1))

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, 0

###########

class ZMnbinom:
    """
    Zero-modified negative binomial distribution.

    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param __p0: base RV probability in zero.
    :type __p0: ``float``
    :param __temp: base RV distribution.
    :type __temp: ``scipy.stats._discrete_distns.nbinom_gen``
    :param __ZTtemp: correspondent ZT distribution.

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          ZM negative binomial distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          ZM negative binomial mixing parameter.
    """
    name = 'ZMnbinom'
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.a=1 - self.p
        self.b=(self.n - 1) * (1 - self.p)
        self.__p0 = np.array([(1 / self.p) ** -self.n])
        self.p0M = kwargs['p0M']
        # self.name = 'ZMnbinom'
        self.__temp = stats.nbinom(n=self.n, p=self.p)
        self.__tempZT=ZTnbinom(n=self.n,p=self.p)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, var):
        assert isinstance(float(var), float) and var >= 0, logger.error(
            'The Zero-Modified Negative Binomial parameter n must be a positive number')
        self.__n = float(var)

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Modified Negative Binomial parameter p must be between zero and one.')
        self.__p = var

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Modified Negative Binomial parameter p0M must be between zero and one.')
        self.__p0M = var

    def pmf(self, k):
        """
        Probability mass function

        :param k: probability mass function will be computed in k.
        :type k: int

        :return: pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return (self.__temp.pmf(k) * (1 - self.p0M)) / (1 - self.__p0)

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative density function will be computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__p0M + (1 - self.__p0M) * (self.__temp.cdf(k) - self.__temp.cdf(0)) / (1 - self.__temp.cdf(0))

    # def __ZTrvs(self, n):
    #     n = int(n)
    #     distTemp = stats.nbinom(n=self.n,p=self.p)
    #     p0 = distTemp.pmf(0)
    #     nsim = int(n + np.ceil(1.10 * p0 * n))  # non rigoroso
    #     r_ = distTemp.rvs(nsim)
    #     r_ = r_[r_ > 0]
    #     while r_.shape[0] < n:
    #         l_ = int(np.maximum(np.ceil(1.10 * (n - r_.shape[0]) * p0), 10))  # non rigoroso
    #         r_ = np.concatenate((r_, distTemp.rvs(l_)))
    #         r_ = r_[r_ > 0]
    #     r_ = r_[:n]
    #     return (r_)

    def rvs(self, size,random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``int`` or ``numpy.ndrarry``

        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')
        r_= stats.bernoulli(p=1-self.p0M).rvs(size,random_state=random_state)
        c_= np.where(r_==1)[0]
        if int(len(c_)) > 0:
            r_[c_]= self.__tempZT.rvs(int(len(c_)))
        return r_
    # def rvs(self, n):
    #     r_ = stats.bernoulli(p=1 - self.p0M).rvs(int(n))
    #     c_ = np.where(r_ == 1)[0]
    #     if int(len(c_)) > 0:
    #         r_[c_] = self.__ZTrvs(int(len(c_)))
    #     return r_

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :type q:``numpy.float64`` or ``numpy.ndarray``
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        p_ = np.array(q)
        temp = self.__temp.ppf((1 - self.__temp.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.__temp.cdf(0))
        return temp
        # zeros = np.where(p_ <= self.p0M)[0]
        # if zeros.size == 0:
        #     return temp
        # else:
        #     if p_.shape == ():
        #         temp = self.p0M
        #     else:
        #         temp[zeros] = self.p0M
        #     return temp
        # return temp

    def pgf(self, f):
        """
        It computes the probability generating function of the RV given the (a,b,k) parametrization.

        :return: probability generated in f.
        :rtype: ``numpy.ndarray``

        """
        return self.p0M+(1-self.p0M)*((1/(1-(f-1)*self.a/(1-self.a)))**(self.b/self.a+1)-(1-self.a)**(self.b/self.a+1))/(1-(1-self.a)**(self.b/self.a+1))

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, self.p0M#+(1-self.p0M)*self.__temp.pmf(0)

#############

class ZMlogser:
    """
    Zero-modified discrete logrithmic distribution.

    :param a: RV a parameter according to the (a,b,k) parametrization.
    :type a: ``float``
    :param b: RV b parameter according to the (a,b,k) parametrization.
    :type b: ``float``
    :param __p0: base RV probability in zero.
    :type __p0: ``float``
    :param __temp: base RV distribution.
    :type __temp: ``scipy.stats._discrete_distns.logser_gen``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          ZM discrete logarithmic distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          ZM discrete logarithmic mixing parameter.
    """
    name = 'ZMlogser'
    def __init__(self, **kwargs):
        self.p = kwargs['p']
        self.__p0 = 0
        self.p0M = kwargs['p0M']
        # self.name = 'ZMlogser'
        self.__temp = stats.logser(p=self.p)

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Modified Geometric distribution parameter p must be between zero and one.')
        self.__p = var

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, var):
        assert var >= 0 and var <= 1, logger.error(
            'The Zero-Modified Geometric parameter p0M must be between zero and one.')
        self.__p0M = var

    def pmf(self, x):
        return (self.__temp.pmf(x) * (1 - self.__p0M)) / (1 - self.__p0)

    def cdf(self, k):
        # da capire bene: p0M o probabilità in zero?
        return self.__p0M + (1 - self.__p0M) * (self.__temp.cdf(k) - self.__temp.cdf(0)) / (1 - self.__temp.cdf(0))

    def rvs(self, size,random_state=None):

        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')
        np.random.seed(random_state)
        q_ = np.random.uniform(0, 1, size)
        return self.ppf(q_)

    def ppf(self, p_):
        p_ = np.array(p_)
        temp = self.__temp.ppf((1 - self.__temp.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.__temp.cdf(0))

        zeros = np.where(p_ <= self.p0M)[0]
        if zeros.size == 0:
            return temp
        else:
            if p_.shape == ():
                temp = self.p0M
            else:
                temp[zeros] = self.p0M
            return temp
        return temp

###### Exponential distribution

class Exponential:
    """
    expontential distribution.

    :param theta: exponential distribution theta parameter.
    :type theta: ``float``
    :param loc: location parameter
    :type loc: ``float``

    """
    name='exponential'
    def __init__(self, loc=0, theta=1):
        self.theta = theta
        self.loc=loc
        #scipy exponential distribution
        self.dist = stats.expon(loc=self.loc)

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, var):
        assert var > 0 , logger.error(
            'The exponential distribution parameter must be strictly bigger than zero')
        self.__theta = var

    def pdf(self,x):
        """
        Probability density function.

        :param x: the probability density function will be computed in x.
        :type x:``numpy.ndarray``
        :return: pdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.theta*self.dist.pdf(self.loc+self.theta*x)
        # try:
        #     x=np.array(x)
        #     assert isinstance(x, np.ndarray), logger.error('x values must be an array')
        # except:
        #     logger.error('Please provide the x values you want to evaluate as an array')
        #
        # temp= self.theta*np.exp(-self.theta*x)
        #
        # zeros = np.where(x < 0)[0]
        # if zeros.size == 0:
        #     return temp
        # else:
        #     if x.shape == ():
        #         temp = 0.
        #     else:
        #         temp[zeros] = 0.0
        #     return temp

    def logpdf(self,x):
        """
        Log of the probability density function.

        :param x: the log of the probability function will be computed in x.
        :type x:``numpy.ndarray``
        :return: logpdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.logpdf(self.theta*x)

    def cdf(self,x):
        """
        Cumulative distribution function.

        :param x: the cumulative distribution function will be computed in x.
        :type x: ``numpy.ndarray``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.cdf(self.loc+self.theta * (x-self.loc))
        #
        # try:
        #     x=np.array(x)
        #     assert isinstance(x, np.ndarray), logger.error('x values must be an array')
        # except:
        #     logger.error('Please provide the x quantiles you want to evaluate as an array')
        #
        # temp= 1-np.exp(-self.theta*x)
        #
        # zeros = np.where(x <= 0)[0]
        # if zeros.size == 0:
        #     return temp
        # else:
        #     if x.shape == ():
        #         temp = 0.
        #     else:
        #         temp[zeros] = 0.0
        #     return temp

    def logcdf(self,x):
        """
        Log of the cumulative distribution function.

        :param x: log of the cumulative density function computed in k.
        :type x: ``int``
        :return: cdf`
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.logcdf(self.theta*x)

    def sf(self,x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in k.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.sf(self.theta*x)

    def logsf(self,x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.logsf(self.theta*x)

    def isf(self,x):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :type q: ``numpy.ndarray``
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.isf(self.theta*x)

    # def fit(self,x):
    #     # try:
    #     #     x=np.array(x)
    #     #     assert isinstance(x, np.ndarray), logger.error('x values must be an array')
    #     #
    #     #     if len(x.shape) != 1:
    #     #         x=x.flatten()
    #     #         logger.info('The vector is flattened to be one-dimensional')
    #     # except:
    #     #     logger.error('Please provide the x quantiles you want to evaluate as an array')
    #     #
    #     # self.theta = 1/np.mean(x)

    def rvs(self, size,random_state=42):
        """
        Random variates.

        :param size: random variates sample size.
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
        return stats.expon.rvs(size=size,random_state=random_state)/self.theta+self.loc

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
        return 1/ self.theta

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

    def ppf(self,x):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :type q: ``numpy.ndarray``
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        try:
            x=np.array(x)
            assert isinstance(x, np.ndarray), logger.error('x values must be an array')
        except:
            logger.error('Please provide the x quantiles you want to evaluate as an array')

        temp= -np.log(1-x)/self.theta

        zeros = np.where(((x>=1.)& (x <=0.)))[0]

        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = np.float64('nan')
            else:
                temp[zeros] = np.float64('nan')
            return temp

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v])
        out = (1 - np.exp(-self.theta * v)) / self.theta
        out[v < 0] = v[v < 0]
        return out

    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        return (1 - stats.expon.cdf(self.theta * (d - loc)))

## gamma
class Gamma:
    """
    Wrapper to scipy gamma distribution.
    When a is an integer it reduces to an Erling distribution.
    When a=1 it reduces to an Exponential distribution.

    :param a: gamma shape parameter a.
    :type a: ``int`` or ``float``
    :param beta: inverse of the gamma scale parameter.
    :type beta: ``float``
    :param scale: gamma scale parameter.
    :type scale: ``float``
    :param loc: gamma location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.gamma_gen ``
    """

    def __init__(self,loc=0,scale=1,**kwargs):
        self.a=kwargs['a']
        self.scale = scale
        self.loc = loc
        self.__dist=stats.gamma(a=self.a,loc=self.loc,scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: int
        :param random_state: random state for the random number generator.
        :type random_state: int

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size,random_state=random_state)

    def pdf(self,x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: int

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self,x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: int
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self,x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self,x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self,x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in k.
        :type x: int
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self,x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in k.
        :type x: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self,q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self,q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self,n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self,data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)


    def expect(self,func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self,alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self,v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v])
        try:
            beta = 1 / self.scale
        except:
            beta = 1
        alpha = self.a
        out = (alpha / beta) * special.gammainc(alpha + 1, beta * v) + v * (
                1 - special.gammainc(alpha, beta * v))
        out[v < 0] = v[v < 0]
        return out

    def den(self,d,loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            beta = 1 / self.spar['scale']
        except:
            beta = 1

        return 1 - stats.gamma(loc=0,a=self.a,scale=1/beta).cdf(d - loc)

## invgamma

class Invgamma:
    """
    Wrapper to scipy inverse gamma distribution.

    :param a: gamma shape parameter a.
    :type a: ``int`` or ``float``
    :param scale: gamma scale parameter.
    :type scale: ``float``
    :param loc: gamma location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.gamma_gen ``
    """

    def __init__(self,loc=0,scale=1,**kwargs):
        self.a=kwargs['a']
        self.scale = scale
        self.loc = loc
        self.__dist=stats.invgamma(a=self.a,
                                   loc=self.loc,
                                   scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: int
        :param random_state: random state for the random number generator.
        :type random_state: int

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size,random_state=random_state)

    def pdf(self,x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: int

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self,x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: int
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self,x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self,x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self,x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in k.
        :type x: int
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self,x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in k.
        :type x: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self,q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self,q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self,n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self,data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)


    def expect(self,func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self,alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self,v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v]).flatten()
        out=v.copy()
        out[v>0]=v[v>0]*special.gammainc(self.a,self.scale/v[v>0])+self.scale*(1-special.gammainc(self.a-1,self.scale/v[v>0]))*special.gamma(self.a-1)/special.gamma(self.a)
        return out

    def den(self,d,loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        return 1 - stats.invgamma(loc=0,a=self.a,scale= self.scale).cdf(d - loc)

## genpareto

class Genpareto:
    """
    Wrapper to scipy genpareto distribution.
    When c=0 it reduces to an Exponential distribution.
    When c=-1 it reduces to a uniform distribution.
    When the correct parametrization is adopted, it is possible to fit all the Pareto types.

    :param c: genpareto shape parameter c.
    :type c: `` float``
    :param scale: genpareto scale parameter.
    :type scale: ``float``
    :param loc: genpareto location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.genpareto_gen ``

    """

    def __init__(self, loc=0,scale=1, **kwargs):
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc
        self.__dist = stats.genpareto(c=self.c, loc=self.loc, scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v])
        try:
            scale_ = self.scale
        except:
            scale_ = 1
        out = (scale_ / (self.c - 1)) * ((1 + self.c * v / scale_) ** (1 - 1 / self.c) - 1)
        out[v < 0] = v[v < 0]
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            scale_ = self.scale
        except:
            scale_ = 1

        return 1 - stats.genpareto(loc=0,c=self.c,scale=scale_).cdf(d - loc)

## lognormal

class Lognorm:
    """
    Wrapper to scipy lognormal distribution.

    :param s: lognormal shape parameter s.
    :type s: `` float``
    :param scale: lognormal scale parameter.
    :type scale: ``float``
    :param loc: lognormal location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.lognorm_gen ``


    """

    def __init__(self, loc=0, scale=1, **kwargs):
        self.s = kwargs['s']
        self.scale = scale
        self.loc = loc
        self.__dist = stats.lognorm(s=self.s, loc=self.loc, scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v])
        out = v.copy()
        try:
            loc = np.log(self.scale)
        except:
            loc = 0
        shape = self.s
        out[v > 0] = np.exp(loc + shape ** 2 / 2) * (stats.norm.cdf((np.log(v[v > 0]) - (loc + shape ** 2)) / shape)) + \
                     v[v > 0] * (
                             1 - stats.norm.cdf((np.log(v[v > 0]) - loc) / shape))
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            loc_ = np.log(self.scale)
        except:
            loc_ = 0
        return 1 - stats.lognorm(scale=np.exp(loc_), s=self.s).cdf(d - loc)

## burr

class Burr12:
    """
    Wrapper to scipy burr distribution.
    It is referred to the Burr Type XII, Singh–Maddala distribution.
    When d=1, this is a Fisk distribution.
    When c=d, this is a Paralogistic distribution.

    :param c: burr shape parameter c.
    :type c: `` float``
    :param d: burr shape parameter d.
    :type d: `` float``
    :param scale: burr scale parameter.
    :type scale: ``float``
    :param loc: burr location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.lognorm_gen ``


    """

    def __init__(self, loc=0, scale=1, **kwargs):
        self.c = kwargs['c']
        self.d = kwargs['d']
        self.scale = scale
        self.loc = loc
        self.__dist = stats.burr12(c=self.c,
                                    d=self.d,
                                    loc=self.loc,
                                    scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v]).flatten()
        out = v.copy()
        u = v.copy()
        u[v>0] = 1 / (1 + (v[v>0] / self.scale) ** self.c)
        out[v>0] = v[v>0]*(u[v>0]**self.d)+special.betaincinv(1+1/self.c,self.d-1/self.c,1-u[v>0])*(self.scale*special.gamma(1+1/self.c)*special.gamma(self.d-1/self.c)/special.gamma(self.d))
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """

        return 1 - stats.burr12(c=self.c,d=self.d,scale=self.scale).cdf(d - loc)

## dagum

class Dagum:
    """
    Wrapper to scipy mielke distribution.
    It is referred to the Inverse Burr, Mielke Beta-Kappa.
    When d=s, this is an inverse paralogistic.

    :param d: dagum shape parameter d.
    :type d: `` float``
    :param s: dagum shape parameter s.
    :type s: `` float``
    :param scale: dagum scale parameter.
    :type scale: ``float``
    :param loc: dagum location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.lognorm_gen ``


    """

    def __init__(self, loc=0, scale=1, **kwargs):
        self.d = kwargs['d']
        self.s = kwargs['s']
        self.k=self.d*self.s
        self.scale = scale
        self.loc = loc
        self.__dist = stats.mielke(k=self.k,
                                    s=self.s,
                                    loc=self.loc,
                                    scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v]).flatten()
        out=v.copy()
        u=v.copy()
        u[v>0] = (v[v>0] / self.scale) ** self.s / (1 + (v[v>0] / self.scale) ** self.s)
        out[v>0] = v[v>0]*(1-u[v>0]**(self.d))+special.betaincinv(self.d+1/self.s,1-1/self.s,u[v>0])*(self.scale*special.gamma(self.d+1/self.s)*special.gamma(1-1/self.s)/special.gamma(self.d))
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """

        return 1 - stats.mielke(k=self.k,s=self.s,scale=self.scale).cdf(d - loc)

## weinbull

class Weibull_min:
    """
    Wrapper to scipy weinbull distribution.

    :param c: dagum shape parameter c.
    :type c: ``float``
    :param scale: dagum scale parameter.
    :type scale: ``float``
    :param loc: dagum location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.weinbullmin_gen``

    """

    def __init__(self, loc=0, scale=1, **kwargs):
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc
        self.__dist = stats.weibull_min(c=self.c,
                                    loc=self.loc,
                                    scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v]).flatten()
        out=v.copy()
        out[v>0]=v[v>0]*np.exp(-(v[v>0]/self.scale)**self.c)+self.scale*special.gamma(1+1/self.c)*special.gammainc(1+1/self.c,(v[v>0]/self.scale)**self.c)
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """

        return 1 - stats.weibull_min(c=self.c,scale=self.scale).cdf(d - loc)


## inverse weibull

class Invweibull:
    """
    Wrapper to scipy inverse weinbull distribution.

    :param c: inverse weinbull shape parameter c.
    :type c: ``float``
    :param scale: inverse weinbull scale parameter.
    :type scale: ``float``
    :param loc: inverse weinbull location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.weinbullmin_gen``

    """

    def __init__(self, loc=0, scale=1, **kwargs):
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc
        self.__dist = stats.invweibull(c=self.c,
                                    loc=self.loc,
                                    scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v]).flatten()
        out=v.copy()
        out[v>0]=v[v>0]*(1-np.exp(-(self.scale/v[v>0])**self.c))+self.scale*special.gamma(1-1/self.c)*(1-special.gammainc(1-1/self.c,(self.scale/v[v>0])**self.c))
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """

        return 1 - stats.invweibull(c=self.c,scale=self.scale).cdf(d - loc)

# beta

class Beta:
    """
    Wrapper to scipy beta distribution.

    :param a: beta shape parameter a.
    :type a: ``float``
    :param b: beta shape parameter b.
    :type b: ``float``
    :param scale: beta scale parameter.
    :type scale: ``float``
    :param loc: beta location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.beta_gen``


    """

    def __init__(self, loc=0, scale=1, **kwargs):
        self.a = kwargs['a']
        self.b = kwargs['b']
        self.scale = scale
        self.loc = loc
        self.__dist = stats.beta(a=self.a,
                                    b=self.b,
                                    loc=self.loc,
                                    scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v]).flatten()
        out=v.copy()
        u=v/self.scale
        out[v>0]=v[v>0]*(1-special.betaincinv(self.a,self.b,u[v>0]))+special.betaincinv(self.a+1,self.b,u[v>0])*self.scale*special.gamma(self.a+self.b)*special.gamma(self.a+1)/(special.gamma(self.a+self.b+1)*special.gamma(self.a))
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """

        return 1 - stats.beta(loc=0,a=self.a,b=self.b,scale=self.scale).cdf(d - loc)

# special.gammaincc * gamma(a)
