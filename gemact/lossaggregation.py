from .libraries import *
from . import config
from . import helperfunctions as hf
from . import copulas as copulas
from . import distributions as distributions
from .calculators import AEPCalculator, MCCalculator

quick_setup()
logger = log.name('lossaggregation')


class Margins:
    """
    Marginal components of Loss Aggregation.

    :param dist: list of the marginal distributions.
    :type dist: ``list``
    :param par: list of the marginal distributions parameters. It must be a list of dictionaries.
    :type par: ``list``
    """
    
    def __init__(
        self,
        dist,
        par
        ):
        self.dist = dist
        self.par = par
        self.dim = len(self.dist)
    
    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, value):
        hf.assert_type_value(value, 'par', logger, type=(list))
        hf.check_condition(
            len(value), len(self.dist), 'par length', logger
        )
        
        for j in range(len(value)):
            hf.assert_type_value(value[j], 'par item', logger, type=(dict))
            
            try:
                eval(config.DIST_DICT[self.dist[j]])(**value[j])
            except Exception:
                logger.error('Please make sure that marginal %s is correctly parametrized.\n See %s' % (j+1, config.SITE_LINK))
                raise
        self.__par = value

    @property
    def dist(self):
        return self.__dist

    @dist.setter
    def dist(self, value):
        hf.assert_type_value(value, 'dist', logger, type=(list))
        hf.check_condition(len(value), 1, 'margins length', logger, '>')

        for j in range(len(value)):
            hf.assert_member(value[j], config.DIST_DICT, logger, config.SITE_LINK)
            hf.assert_member('severity', eval(config.DIST_DICT[value[j]]).category(), logger, config.SITE_LINK)
        self.__dist = value

    def model(self, m):
        return eval(config.DIST_DICT[self.dist[m]])(**self.par[m])

    def ppf(self, q):
        """
        Margin percent point function, a.k.a. the quantile function,
        inverse of the cumulative distribution function.

        :param q: probabilities. Shape must be (dim, size) where size is the number of points to be evaluated.
        :type q: ``numpy.ndarray``
        :return: quantile.
        :rtype: ``numpy.ndarray``
        """
        result = [self.model(j).ppf(q[j, :]) for j in range(self.dim)]
        return np.array(result)

    def cdf(self, x):
        """
        Margin cumulative distribution function.

        :param x: quantiles where the cumulative distribution function is evaluated.
                Shape must be (dim, size) where size is the number of points to be evaluated.
        :type x: ``numpy.ndarray``
        :return: cumulative distribution function.
        :rtype: ``numpy.ndarray``
        """
        result = [self.model(j).cdf(x[j, :]) for j in range(self.dim)]
        return np.array(result)


class Copula:
    """
    Copula component of Loss Aggregation.

    :param dist: name of the copula distribution.
    :type dist: ``str``
    :param par: parameters of the copula distribution.
    :type par: ``dict``
    """

    def __init__(
        self,
        dist,
        par,
        ):
        self.dist = dist
        self.par = par

    @property
    def dist(self):
        return self.__dist

    @dist.setter
    def dist(self, value):
        hf.assert_type_value(value, 'dist', logger, type=(str))
        hf.assert_member(value, config.COP_DICT, logger, config.SITE_LINK)
        self.__dist = value

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, value):
        hf.assert_type_value(value, 'par', logger, type=(dict))
        try:
            eval(config.COP_DICT[self.dist])(**value)
        except Exception:
            logger.error('Copula not correctly parametrized.\n See %s' % config.SITE_LINK)
            raise
        self.__par = value

    @property
    def model(self):
        return eval(config.COP_DICT[self.dist])(**self.par)

    @property
    def dim(self):
        return self.model.dim

    def rvs(self, size, random_state):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``
        :return: random variates.
        :rtype: ``numpy.ndarray``
        """
        result = self.model.rvs(size, random_state)
        return np.array(result)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantiles where the cumulative distribution function is evaluated.
                Shape must be (dim, size) where size is the number of points to be evaluated.
        :type x: ``numpy.ndarray``
        :return: cumulative distribution function.
        :rtype: ``numpy.ndarray``
        """
        result = self.model.cdf(x.transpose())
        return np.array(result)


class LossAggregation:
    """
    Class representing the sum of positive random variables.
    Dependence structure is specified by a copula.

    :param copula: name of the copula that describes the dependence structure.
    :type copula: ``Copula``
    :param margins: list of the marginal distributions.
    :type margins: ``Margins``
    :param n_sim: number of Monte Carlo simulations (optional). If ``None`` the simulation is skipped.
    :type n_sim: ``int``
    :param random_state: random state for the random number generator (optional).
    :type random_state: ``int``
    :param n_iter: number of AEP algorithm iterations (optional).
    :type n_iter: ``int``
    :param tol: tolerance threshold for AEP ppf, maximum allowed absolute difference between cumulative probability values (optional).
    :type tol: ``float``
    """

    def __init__(
        self,
        copula,
        margins,
        n_sim=None,
        random_state=None,
        n_iter=7,
        tol=1e-4
        ):
        self.copula = copula
        self.margins = margins
        self.n_sim = n_sim
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        self.__dist = [None]
        self.dist_calculate()
        
    @property
    def margins(self):
        return self.__margins

    @margins.setter
    def margins(self, value):
        hf.assert_type_value(value, 'margins', logger, type=(Margins))
        self.__margins = value

    @property
    def copula(self):
        return self.__copula

    @copula.setter
    def copula(self, value):
        hf.assert_type_value(value, 'copula', logger, type=(Copula))
        self.__copula = value

    @property
    def n_iter(self):
        return self.__n_iter

    @n_iter.setter
    def n_iter(self, value):
        hf.assert_type_value(
            value, 'n_iter', logger,
            type=(int, float),
            lower_bound=1, lower_close=True
            )
        value = int(value)
        self.__n_iter = value

    @property
    def n_sim(self):
        return self.__n_sim

    @n_sim.setter
    def n_sim(self, value):
        if value is not None:
            hf.assert_type_value(
                value, 'n_sim', logger,
                type=(int, float),
                lower_bound=1, lower_close=False
                )
            value = int(value)
        self.__n_sim = value

    @property
    def random_state(self):
        return self.__random_state

    @random_state.setter
    def random_state(self, value):
        self.__random_state = hf.handle_random_state(value, logger)

    @property
    def tol(self):
        return self.__tol

    @tol.setter
    def tol(self, value):
        hf.assert_type_value(
            value, 'tol', logger,
            type=(float, np.floating),
            lower_bound=0, lower_close=False,
            upper_bound=0.5, upper_close=True
            )
        self.__tol = value

    @property
    def dist(self):
        return self.__dist

    @dist.setter
    def dist(self, value):
        hf.assert_type_value(
            value, 'dist', logger, distributions.PWC
        )
        self.__dist = value

    def dist_calculate(
            self,
            n_sim=None,
            random_state=None
            ):
        """
        Approximate the distribution of the sum of random variable with a
        given dependence structure by executing a Monte Carlo simulation.
        The resulting distribution can be accessed via the ``dist`` property, which is a ``distributions.PWC`` object.
        
        :param n_sim: number of simulations of Monte Carlo simulation (optional).
        :type n_sim: ``int``
        :param random_state: random state for the random number generator (optional).
        :type random_state: ``int``
        :return: Void.
        :rtype: ``None``
        """
        
        # check if n_sim is None
        if (n_sim is None and self.n_sim is None):
            logger.warning('Monte Carlo simulation is omitted as n_sim is missing.')
            return
        
        if n_sim is not None:
            self.n_sim = n_sim

        if random_state is not None:
            self.random_state = random_state

        nodes, cumprobs = MCCalculator.simulation_execute(
            size=self.n_sim, random_state=self.random_state,
            copula=self.copula, margins=self.margins
            )

        self.dist = distributions.PWC(
            nodes=nodes,
            cumprobs=cumprobs
        )
        return

    def cdf(self, x, method='mc', n_iter=None):
        """
        Cumulative distribution function of the random variable sum.
        If ``method`` is Monte Carlo ('mc') the function relies on the approximated distribution
        calculated via ``dist_calculate`` method when the object is initiated (accessed via the ``dist`` property).
        If ``method`` is AEP ('aep') the function is evaluated pointwise, on-the-fly, regardless of the ``dist`` property.
        
        :param x: quantiles where the cumulative distribution function are evaluated.
        :type x: ``float``
        :param method: method to approximate the cdf of the sum of the random variables.
                        One of AEP ('aep') and Monte Carlo simulation ('mc').
        :type method: ``string``
        :param n_iter: number of AEP algorithm iterations (optional).
        :type n_iter: ``int``

        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        
        hf.assert_member(
            method,
            config.LOSS_AGGREGATION_METHOD,
            logger
        )
        
        hf.assert_type_value(x, 'x', logger, (int, float, np.ndarray, list)) 
        if method == 'aep':
            hf.check_condition(
                self.copula.dim, config.DCEILING, 'Copula dimension for AEP', logger, type='<='
            )
            if n_iter is not None:
                self.n_iter = n_iter
            return AEPCalculator.cdf(
                x=x, n_iter=self.n_iter, copula=self.copula, margins=self.margins
                )
        else:
            if self._check_missing_dist():
                return None
            else:
                return self.dist.cdf(x)
            
    def sf(self, x, method='mc', n_iter=None):
        """
        Survival function of the random variable sum.
        If ``method`` is Monte Carlo ('mc') the function relies on the approximated distribution
        calculated via ``dist_calculate`` method when the object is initiated (accessed via the ``dist`` property).
        If ``method`` is AEP ('aep') the function is evaluated pointwise, on-the-fly, regardless of the ``dist`` property.
        
        :param x: quantiles where the survival function are evaluated.
        :type x: ``float``
        :param method: method to approximate the survival function of the sum of the random variables.
                        One of AEP ('aep') and Monte Carlo simulation ('mc').
        :type method: ``string``
        :param n_iter: number of AEP algorithm iterations (optional).
        :type n_iter: ``int``
        :return: survival function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return 1 - self.cdf(x, method, n_iter)

    def ppf(self, q, method='mc', n_iter=None, tol=1e-4):
        """
        Percent point function, a.k.a. the quantile function, of the random variable sum.
        Inverse of cumulative distribution function.
        If ``method`` is Monte Carlo ('mc') the function relies on the approximated distribution
        calculated via ``dist_calculate`` method when the object is initiated (accessed via the ``dist`` property).
        If ``method`` is AEP ('aep') the function is evaluated pointwise, on-the-fly, regardless of the ``dist`` property.
        It adopts the ``scipy.optimize.brentq`` optimizer.
        
        :param q: probabilities where point function is evaluated.
        :type q: ``float``, ``numpy.ndarray``, ``numpy.floating``
        :param method: method to approximate the ppf of the sum of the random variables.
                        One of AEP ('aep') and Monte Carlo simulation ('mc').
        :type method: ``string``
        :param n_iter: number of AEP algorithm iterations (optional).
        :type n_iter: ``int``
        :param tol: tolerance threshold, maximum allowed absolute difference between cumulative probability values (optional).
        :type tol: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        hf.assert_member(
            method,
            config.LOSS_AGGREGATION_METHOD,
            logger
        )
        
        if method == "aep":
            hf.check_condition(
                self.copula.dim, config.DCEILING, 'Copula dimension for AEP', logger, type='<='
            )
            if n_iter is not None:
                self.n_iter = n_iter
            return AEPCalculator.ppf(
                q, self.n_iter, self.copula, self.margins, tol
                )
        else:
            if self._check_missing_dist():
                return None
            else:
                return self.dist.ppf(q)

    def rvs(self, size=1, random_state=None, method='mc', n_iter=None, tol=None):
        """
        Random variates generator function.
        If ``method`` is Monte Carlo ('mc') the function use inverse transform sampling
        via copula, margins and then apply the sum.
        If ``method`` is AEP ('aep') the function use the inverse transform sampling via 'aep' ppf.
        The latter option is time demanding.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``
        :param method: method to execute the generator of random variates.
                        One of AEP ('aep') and Monte Carlo simulation ('mc').
        :type method: ``string``
        :param n_iter: number of AEP algorithm iterations (optional).
        :type n_iter: ``int``
        :param tol: tolerance threshold, maximum allowed absolute difference between cumulative probability values (optional).
        :type tol: ``float``
        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        hf.assert_type_value(
            value=size,
            name='size',
            logger=logger,
            type=(int, float),
            lower_bound=1
            )
        size = int(size)
        
        random_state = hf.handle_random_state(random_state)

        if method == "aep":
            hf.check_condition(
                self.copula.dim, config.DCEILING, 'Copula dimension for AEP', logger, type='<='
            )
            if size >= 30:
                logger.warning(
                    'AEP based rvs is time demanding. For a faster execution please rely on MC method (method="mc").'
                ) 
            if n_iter is not None:
                self.n_iter = n_iter
            if tol is not None:
                self.tol = tol
            return AEPCalculator.rvs(
                size, random_state, self.n_iter, self.copula, self.margins, tol
                )
        else:
            return MCCalculator.rvs(
                size, random_state, self.copula, self.margins
                )
            
    def moment(self, central=False, n=1):
        """
        Moment of order n of the random variable sum.
        It is based on Monte Carlo simulation results, i.e. ``dist`` property.

        :param central: ``True`` if the moment is central, ``False`` if the moment is raw.
        :type central: ``bool``
        :param n: order of the moment, optional (default is 1).
        :type n: ``int``
        :return: moment of order n.
        :rtype: ``numpy.float64``
        """
        if self._check_missing_dist():
            return None
        else:
            return self.dist.moment(central, n)

    def mean(self):
        """
        Mean of the random variable sum.
        It is based on Monte Carlo simulation results, i.e. ``dist`` property.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        if self._check_missing_dist():
            return None
        else:
            return self.dist.mean()
    
    def skewness(self):
        """
        Skewness of the random variable sum.
        It is based on Monte Carlo simulation results, i.e. ``dist`` property.

        :return: skewness.
        :rtype: ``numpy.float64``
        """
        if self._check_missing_dist():
            return None
        else:
            return self.dist.skewness()

    def var(self):
        """
        Variance of the random variable sum.
        It is based on Monte Carlo simulation results, i.e. ``dist`` property.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        if self._check_missing_dist():
            return None
        else:
            return self.dist.var()

    def std(self):
        """
        Standard deviation of the random variable sum.
        It is based on Monte Carlo simulation results, i.e. ``dist`` property.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        if self._check_missing_dist():
            return None
        else:
            return self.dist.std()

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``int``, ``float``, ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        return self.dist.lev(v=v)

    def censored_moment(self, n, u, v):
        """
        Non-central moment of order n of the transformed random variable min(max(x - u, 0), v).
        When n = 1 it is the so-called stop loss transformation function.
                
        :param u: lower censoring point.
        :type u: ``int``, ``float``, ``numpy.float`` or ``numpy.ndarray``
        :param v: difference between the upper and the lower censoring points, i.e. v + u is the upper censoring point.
        :type v: ``int``, ``float``, ``numpy.float`` or ``numpy.ndarray``
        :param n: moment order.
        :type n: ``int``
        :return: censored raw moment of order n.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        return self.dist.censored_moment(n=n, u=u, v=v)

    def plot_cdf(self, log_x_scale=False, log_y_scale=False, **kwargs):
        """
        Plot the cumulative distribution function of the random variable sum.
        It is based on Monte Carlo simulation results, i.e. ``dist`` property.

        :type idx: ``int``
        :param log_x_scale: if ``True`` the x-axis scale is logarithmic (optional).
        :type log_x_scale: ``bool``
        :param log_y_scale: if ``True`` the y-axis scale is logarithmic (optional).
        :type log_y_scale: ``bool``
        :param \\**kwargs:
            Additional parameters as those for ``matplotlib.axes.Axes.step``.
        :return: plot of the cdf.
        :rtype: ``matplotlib.figure.Figure``
        """

        if self._check_missing_dist():
            return
        hf.assert_type_value(log_x_scale, 'log_x_scale', logger, bool)
        hf.assert_type_value(log_y_scale, 'log_y_scale', logger, bool)

        x_ = self.dist.nodes
        y_ = self.dist.cumprobs

        figure = plt.figure()
        ax = figure.add_subplot(111)

        ax.step(x_, y_, '-', where='post', **kwargs)
        if log_y_scale:
            ax.set_yscale('log')
        if log_x_scale:
            ax.set_xscale('log')
        ax.set_title('Random variable sum cumulative distribution function')
        ax.set_ylabel('cdf')
        ax.set_xlabel('nodes')
        return ax

    def _check_missing_dist(self):
        """
        Check that the Monte Carlo based distribution is missing.
        Helper method called before executing other methods based on ``dist`` property.

        :param value: 
        :type value: ``string``
        :return: True or False, depending whether ``dist`` is missing (None) or not.
        :rtype: ``bool``
        """
        output = hf.check_none(
            self.dist,
            logger,
            'warning',
            message='Execution stopped. Missing ``dist``, please execute ``dist_calculate`` first.'
            )
        return output