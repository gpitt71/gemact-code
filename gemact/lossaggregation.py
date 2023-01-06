from .libraries import *
from . import config
from . import helperfunctions as hf
from . import copulas as copulas
from . import distributions as distributions


quick_setup()
logger = log.name('lossaggregation')

class LossAggregation:
    """
        Class representing the sum of positive countinuous random variables.
        Dependence structure is specified by a copula and a set of given marginals.

        :param copula: Name of the copula that describes the dependence structure.
        :type copula: ``str``
        :param copula_par: Parameters of the copula.
        :type copula_par: ``dict``
        :param margins: List of the marginal distributions.
        :type margins: ``list``
        :param margins_pars: List of the marginal distributions parameters. It must be a list of dictionaries.
        :type margins_pars: ``list``

        :param \\**kwargs:
            See below

        :Keyword Arguments:
            * *random_state* (``int``) --
                Random state for the random number generator in MC.
            * *sample_size* (``int``) --
                Number of simulations of Monte Carlo (mc) method.

    """

    def __init__(self, copula, copula_par, margins, margins_pars, **kwargs):
        self.copula = copula
        self.copula_par = copula_par
        self.margins = margins
        self.margins_pars = margins_pars
        self.random_state = kwargs.get('random_state', int(time.time()))
        self.sample_size = kwargs.get('sample_size', 10000)
        self._dist_calculate()

    @property
    def random_state(self):
        return self.__random_state

    @random_state.setter
    def random_state(self, value):
        self.__random_state = hf.handle_random_state(value, logger)

    @property
    def sample_size(self):
        return self.__sample_size

    @sample_size.setter
    def sample_size(self, value):
        hf.assert_type_value(value, 'sample_size', logger, type=(int, float), lower_bound=1, lower_close=False)
        self.__sample_size = int(value)

    @property
    def margins_pars(self):
        return self.__margins_pars

    @margins_pars.setter
    def margins_pars(self, value):
        hf.assert_type_value(value, 'margins_pars', logger, type=(list))
        hf.check_condition(
            len(value), len(self.margins), 'margins_pars', logger
        )
        
        for j in range(len(value)):
            hf.assert_type_value(value[j], 'margins_pars item', logger, type=(dict))
            
            try:
                eval(config.DIST_DICT[self.margins[j]])(**value[j])
            except Exception:
                logger.error('Please make sure that margin %s is correctly parametrized.\n See %s' % (j+1, config.SITE_LINK))
                raise
        self.__margins_pars = value

    @property
    def margins(self):
        return self.__margins

    @margins.setter
    def margins(self, value):
        hf.assert_type_value(value, 'margins', logger, type=(list))
        hf.assert_type_value(len(value), 'margins length', logger, type=(float, int),
            upper_bound=config.DCEILING)

        for j in range(len(value)):
            hf.assert_member(value[j], config.DIST_DICT, logger, config.SITE_LINK)
            hf.assert_member('severity', eval(config.DIST_DICT[value[j]]).category(), logger, config.SITE_LINK)
        self.__margins = value

    @property
    def copula(self):
        return self.__copula

    @copula.setter
    def copula(self, value):
        hf.assert_type_value(value, 'copula', logger, type=(str))
        hf.assert_member(value, config.COP_DICT, logger, config.SITE_LINK)
        self.__copula = value

    @property
    def copula_par(self):
        return self.__copula_par

    @copula_par.setter
    def copula_par(self, value):
        hf.assert_type_value(value, 'copula_par', logger, type=(dict))
       
        try:
            eval(config.COP_DICT[self.copula])(**value)
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
        return ((self.d + 1) ** self.d) / (special.factorial(self.d) * 2 ** self.d)

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

    @property
    def dist(self):
        return self.__dist

    def _dist_calculate(self):
        """
        Approximation of the distribution by calculating nodes, pdf, and cdf.

        :return: Void.
        :rtype: ``None``
        """
        u_ = self._copula_rvs(self.sample_size, self.random_state).T
        xsim = np.sum(self._margins_ppf(u_), axis=0)

        x_ = np.unique(xsim)
        cdf_ = hf.ecdf(xsim)(x_)

        nodes, cumprobs = hf.nodes_and_probs_adjuster(
            x_, cdf_
        )
        # self.__dist = {'epmf': epmf,
        #         'cdf': cdf,
        #         'nodes': nodes}
        self.__dist = distributions.PWL(
            nodes=nodes,
            cumprobs=cumprobs
        )
        return

    def _private_prop_aep_initiate(self, x):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param x: initial value for the quantile where the cumulative distribution function is evaluated.
        :type x: ``float``
        :return: void
        :rtype: ``None``
        """
        self.__b = np.repeat(0, self.d).reshape(1, self.d)  # Vector b of the AEP algorithm.
        self.__h = np.array([[x]])  # Vector h of the AEP algorithm.
        self.__sn = np.array([1])  # Array of +1,-1, 0 indicating whether a volume must be summed,
        # subtracted or ignored, respectively.
        self.__vols = 0  # sum of 'volumes' * 'sn' used in AEP iteration

    def _private_prop_aep_delete(self):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        
        :return: void
        :rtype: ``None``
        """
        del self.__b
        del self.__h
        del self.__sn
        del self.__vols

    def _copula_rvs(self, size, random_state):
        """
        Copula random variates generator function.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``
        """
        result = eval(config.COP_DICT[self.copula])(**self.copula_par).rvs(size, random_state)
        return np.array(result)

    def _copula_cdf(self, k):
        """
        Copula cumulative distribution function.

        :param x: quantiles where the cumulative distribution function is evaluated.
        :type x: ``float`` or ``int`` or ``numpy.ndarray``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        result = eval(config.COP_DICT[self.copula])(**self.copula_par).cdf(k.transpose())
        return np.array(result)

    def _margins_ppf(self, k):
        """
        Margin percent point function, a.k.a. the quantile function,
        inverse of the cumulative distribution function.

        :param k: probability.
        :type k: ``float`` or ``numpy.ndarray``
        :return: quantile.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        result = [eval(config.DIST_DICT[self.margins[j]])(**self.margins_pars[j]).ppf(k[j, :]) for j in range(self.d)]
        return np.array(result)

    def _margins_cdf(self, k):
        """
        Margin cumulative distribution function.

        :param k: quantiles where the cumulative distribution function is evaluated.
        :type k: ``float`` or ``int`` or ``numpy.ndarray``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        result = [eval(config.DIST_DICT[self.margins[j]])(**self.margins_pars[j]).cdf(k[j, :]) for j in range(self.d)]
        return np.array(result)

    def _volume_calc(self):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        """
        mat_ = np.expand_dims(self.mat, axis=2)
        h_ = self.a * self.__h
        b_ = np.expand_dims(self.__b.T, axis=0)
        s_ = self.s.reshape(-1, 1)
        v_ = np.hstack((b_ + h_ * mat_))  # np.dstack(zip( (b_ + h_*mat_) ))[0]    # <- WARNING
        c_ = self._copula_cdf(self._margins_cdf(v_)).reshape(-1, self.__b.shape[0])
        result = np.sum(c_ * (s_ * np.sign(h_) ** self.d), axis=0)
        return result

    def _sn_update(self):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        """
        result = np.repeat(self.__sn, self.n_simpleces) * np.tile(self.m, self.__sn.shape[0])
        return result

    def _h_update(self):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        """
        result = (1 - np.tile(self.card, len(self.__h)) * self.a) * np.repeat(self.__h, len(self.card))
        return result

    def _b_update(self):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        """
        mat_ = self.mat[1:, :].transpose()
        h_ = np.repeat(self.__h, self.n_simpleces).reshape(-1, 1)
        times_ = int(h_.shape[0] / mat_.shape[1])
        result = np.repeat(self.__b, self.n_simpleces, 0)
        result = result + self.a * np.tile(h_, (1, self.d)) * np.tile(mat_, times_).transpose()
        return result

    def _aep_cdf(self, x, n_iter):
        """
        AEP algorithm to approximate cdf.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``float``
        :param n_iter: number of algorithm iterations.
        :type n_iter: ``int``

        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

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

    def _mc_cdf(self, x):
        """
        Cumulative distribution function from Monte Carlo simulation.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int`` or ``float``

        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        # x = np.ravel(x)
        # probs_ = np.concatenate(
        #     ([0, 0], self.dist['cdf'], [1, 1])
        #     )
        # nodes_ = np.concatenate(
        #     ([-np.inf, 0],
        #     self.dist['nodes'],
        #     [self.dist['nodes'][-1] + config.TOLERANCE, np.inf])
        #     )
        # cdf = interp1d(nodes_, probs_)
        # return cdf(x)
        return self.dist.cdf(x)

    def cdf(self, x, method="mc", **kwargs):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``float``
        :param method: method to approximate the cdf of the aggregate loss random variable
                        (i.e. the sum of random variables with a dependence structure specified by a copula).
                        One of AEP ('aep') and Monte Carlo simulation ('mc').
        :type method: ``string``

        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        :param \\**kwargs:
            See below

        :Keyword Arguments:
            * *n_iter* (``int``) --
                Number of iteration of AEP algorithm.
        """
        if method not in config.LOSS_AGGREGATION_METHOD:
            raise ValueError("results: method must be one of %r." % config.LOSS_AGGREGATION_METHOD)
        
        hf.assert_type_value(x, 'x', logger, (int, float, np.ndarray, list))
        isscalar = not isinstance(x, (np.ndarray, list)) 
        x = np.ravel(x)

        if method == 'aep':
            output = np.empty(len(x))
            n_iter = kwargs.get('n_iter', 5)
            for i in range(len(output)):
                output[i] = self._aep_cdf(x[i], n_iter)
            if isscalar:
                output = output.item()

        else: # method == 'mc'
            output = self._mc_cdf(x)

        return output
        

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function,
        inverse of cumulative distribution function from Monte Carlo simulation.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``, ``numpy.ndarray``, ``numpy.floating``

        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        # hf.assert_type_value(
        #     q, 'q', logger, (np.floating, int, float, list, np.ndarray)
        #     )
        # isscalar = not isinstance(q, (np.ndarray, list)) 
        # q = np.ravel(q)
        # for item in q:
        #     hf.assert_type_value(item, 'q', logger, (np.floating, np.integer), upper_bound=1, lower_bound=0)
        
        # probs_ = np.concatenate(
        #     ([0, 0], self.dist['cdf'], [1, 1])
        #     )
        # nodes_ = np.concatenate(
        #     ([-np.inf, 0],
        #     self.dist['nodes'],
        #     [self.dist['nodes'][-1] + config.TOLERANCE, np.inf])
        #     )
        # output = interp1d(probs_, nodes_)(q)
        # if isscalar:
        #     return output.item()
        # else:
        #     return output
        return self.dist.ppf(q)

    def moment(self, central=False, n=1):
        """
        Non-central moment of order n.

        :param n: moment order.
        :type n: ``int``

        :return: raw moment of order n.
        :rtype: ``float``
        """
        return self.dist.moment(central, n)

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
        # hf.assert_type_value(size, 'size', logger, type=(int, float), lower_bound=1, lower_close=False)
        # size = int(size)

        # random_state = hf.handle_random_state(random_state, logger)
        # np.random.seed(random_state)

        # output = self.ppf(np.random.uniform(size=size))
        # return output
        return self.dist.rvs(size, random_state)

    def mean(self):
        """
        Mean of the aggregated loss.

        :return: mean of the aggregated loss.
        :rtype: ``numpy.float64``
        """
        return self.dist.mean()
    
    def skewness(self):
        """
        Skewness of the aggregated loss.

        :return: skewness of the aggregated loss.
        :rtype: ``numpy.float64``
        """
        # return self.moment(central=True, order=3) / self.moment(central=True, order=2) ** 3 / 2
        return self.dist.skewness()

    def std(self):
        """
        Standard deviation of the aggregated loss.

        :return: standard deviation of the aggregated loss.
        :rtype: ``numpy.float64``
        """
        return self.dist.std()