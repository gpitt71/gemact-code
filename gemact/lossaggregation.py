from .libraries import *
from . import config
from . import helperfunctions as hf
from . import copulas as copulas
from . import distributions as distributions
from .calculators import LossAggregationCalculator as Calculator

quick_setup()
logger = log.name('lossaggregation')


class Margins:
    """
    Marginal components of Loss Aggregation.

    :param copula: name of the copula distribution.
    :type copula: ``str``
    :param par: parameters of the copula distribution.
    :type par: ``dict``
    """
    
    def __init__(
        self,
        margins,
        pars
        ):
        self.margins = margins
        self.pars = pars
        self.dim = len(self.margins)
    
    @property
    def pars(self):
        return self.__pars

    @pars.setter
    def pars(self, value):
        hf.assert_type_value(value, 'pars', logger, type=(list))
        hf.check_condition(
            len(value), len(self.margins), 'pars length', logger
        )
        
        for j in range(len(value)):
            hf.assert_type_value(value[j], 'pars item', logger, type=(dict))
            
            try:
                eval(config.DIST_DICT[self.margins[j]])(**value[j])
            except Exception:
                logger.error('Please make sure that margin %s is correctly parametrized.\n See %s' % (j+1, config.SITE_LINK))
                raise
        self.__pars = value

    @property
    def margins(self):
        return self.__margins

    @margins.setter
    def margins(self, value):
        hf.assert_type_value(value, 'margins', logger, type=(list))
        # not evaluated here but inside AEP only.
        # hf.check_condition(len(value), config.DCEILING, 'margins length', logger, '<=')

        for j in range(len(value)):
            hf.assert_member(value[j], config.DIST_DICT, logger, config.SITE_LINK)
            hf.assert_member('severity', eval(config.DIST_DICT[value[j]]).category(), logger, config.SITE_LINK)
        self.__margins = value

    def model(self, m):
        return eval(config.DIST_DICT[self.margins[m]])(**self.pars[m])

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

    :param copula: name of the copula distribution.
    :type copula: ``str``
    :param par: parameters of the copula distribution.
    :type par: ``dict``
    """

    def __init__(
        self,
        copula,
        par,
        ):
        self.copula = copula
        self.par = par
        self.model = eval(config.COP_DICT[self.copula])(**self.par)

    @property
    def copula(self):
        return self.__copula

    @copula.setter
    def copula(self, value):
        hf.assert_type_value(value, 'copula', logger, type=(str))
        hf.assert_member(value, config.COP_DICT, logger, config.SITE_LINK)
        self.__copula = value

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, value):
        hf.assert_type_value(value, 'par', logger, type=(dict))
        try:
            eval(config.COP_DICT[self.copula])(**value)
        except Exception:
            logger.error('Copula not correctly parametrized.\n See %s' % config.SITE_LINK)
            raise
        self.__par = value

    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, value):
        self.__model = value

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
    :type copula: ``str``
    :param copula_par: parameters of the copula.
    :type copula_par: ``dict``
    :param margins: list of the marginal distributions.
    :type margins: ``list``
    :param margins_pars: list of the marginal distributions parameters. It must be a list of dictionaries.
    :type margins_pars: ``list``
    :param size: number of simulations for Monte Carlo method (optional). If ``None`` the calculation is skipped.
    :type size: ``int``
    :param n_iter: number of AEP algorithm iterations (optional).
    :type n_iter: ``int``
    :param random_state: random state for the random number generator (optional).
    :type random_state: ``int``
    """

    def __init__(
        self,
        copula,
        # copula_par,
        margins,
        # margins_pars,
        size=None,
        random_state=None
        ):
        self.copula = copula
        # self.copula_par = copula_par
        self.margins = margins
        # self.margins_pars = margins_pars
        self.random_state = random_state
        self.size = size
        self.calc = Calculator(
            copula=self.copula,
            margins=self.margins
        )
        self.dist_calculate()
        
    @property
    def random_state(self):
        return self.__random_state

    @random_state.setter
    def random_state(self, value):
        self.__random_state = hf.handle_random_state(value, logger)

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, value):
        if value is not None:
            hf.assert_type_value(
                value, 'size', logger,
                type=(int, float),
                lower_bound=1, lower_close=False
                )
            value = int(value)
        self.__size = value

    # @property
    # def margins_pars(self):
    #     return self.__margins_pars

    # @margins_pars.setter
    # def margins_pars(self, value):
    #     hf.assert_type_value(value, 'margins_pars', logger, type=(list))
    #     hf.check_condition(
    #         len(value), len(self.margins), 'margins_pars', logger
    #     )
        
    #     for j in range(len(value)):
    #         hf.assert_type_value(value[j], 'margins_pars item', logger, type=(dict))
            
    #         try:
    #             eval(config.DIST_DICT[self.margins[j]])(**value[j])
    #         except Exception:
    #             logger.error('Please make sure that margin %s is correctly parametrized.\n See %s' % (j+1, config.SITE_LINK))
    #             raise
    #     self.__margins_pars = value

    # @property
    # def margins(self):
    #     return self.__margins

    # @margins.setter
    # def margins(self, value):
    #     hf.assert_type_value(value, 'margins', logger, type=(list))
    #     hf.assert_type_value(len(value), 'margins length', logger, type=(float, int),
    #         upper_bound=config.DCEILING)

    #     for j in range(len(value)):
    #         hf.assert_member(value[j], config.DIST_DICT, logger, config.SITE_LINK)
    #         hf.assert_member('severity', eval(config.DIST_DICT[value[j]]).category(), logger, config.SITE_LINK)
    #     self.__margins = value

    @property
    def margins(self):
        return self.__margins

    @margins.setter
    def margins(self, value):
        hf.assert_type_value(value, 'margins', logger, type=(Margins))
        self.__margins = value

    # @property
    # def copula(self):
    #     return self.__copula

    # @copula.setter
    # def copula(self, value):
    #     hf.assert_type_value(value, 'copula', logger, type=(str))
    #     hf.assert_member(value, config.COP_DICT, logger, config.SITE_LINK)
    #     self.__copula = value

    # @property
    # def copula_par(self):
    #     return self.__copula_par

    # @copula_par.setter
    # def copula_par(self, value):
    #     hf.assert_type_value(value, 'copula_par', logger, type=(dict))
       
    #     try:
    #         eval(config.COP_DICT[self.copula])(**value)
    #     except Exception:
    #         logger.error('Copula not correctly parametrized.\n See %s' % config.SITE_LINK)
    #         raise

    #     self.__copula_par = value

    @property
    def copula(self):
        return self.__copula

    @copula.setter
    def copula(self, value):
        hf.assert_type_value(value, 'copula', logger, type=(Copula))
        self.__copula = value

    # @property
    # def d(self):
    #     return len(self.margins)

    # @property
    # def _a(self):
    #     # Alpha parameter of the AEP algorithm.
    #     return 2. / (self.d + 1)

    # @property
    # def _ext(self):
    #     # Probability correction of the AEP
    #     return ((self.d + 1) ** self.d) / (special.factorial(self.d) * 2 ** self.d)

    # @property
    # def _mat(self):
    #     # Matrix of the vectors in the {0,1}**d space.
    #     return hf.cartesian_product(*([np.array([0, 1])] * self.d)).T

    # @property
    # def _n_simpleces(self):
    #     # AEP number of new simpleces received in each step.
    #     return 2 ** self.d - 1

    # @property
    # def _card(self):
    #     # AEP cardinality of the 'mat' matrix.
    #     return np.sum(self._mat, axis=1)[1:]

    # @property
    # def _s(self):
    #     # AEP array of +1 or -1, indicating whether to sum or subtract a volume, respectively.
    #     return (-1) ** (self.d - np.sum(self._mat, axis=1))

    # @property
    # def _m(self):
    #     # Array of +1, -1, 0, indicating whether the new simpleces origined from sn must be summed,
    #     # subtracted or ignored, respectively.
    #     output = self._card.copy()
    #     greater = np.where(output > (1 / self._a))
    #     equal = np.where(output == (1 / self._a))
    #     lower = np.where(output < (1 / self._a))
    #     output[greater] = (-1) ** (self.d + 1 - output[greater])
    #     output[equal] = 0
    #     output[lower] = (-1) ** (1 + output[lower])
    #     return output

    @property
    def dist(self):
        return self.__dist

    @dist.setter
    def dist(self, value):
        hf.assert_type_value(
            value, 'dist', logger, distributions.PWC
        )
        self.__dist = value

    def dist_calculate(self, size=None, random_state=None):
        """
        Approximate the distribution of the sum of random variable with a
        given dependence structure.
        The distribution can be accessed via the ``dist`` property, which is a ``distributions.PWC`` object.
        
        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``
        :return: Void.
        :rtype: ``None``
        """

        if (size is None) and (self.size is None):
            logger.warning('Distribution calculation is omitted as size is missing')
            self.__dist = None
            return

        if size is not None:
            self.size = size
        if random_state is not None:
            self.random_state = random_state

        nodes, cumprobs = self.calc.mc.dist_calculate(size, random_state)

        self.dist = distributions.PWC(
            nodes=nodes,
            cumprobs=cumprobs
        )
        return

    # def dist_calculate(self, size=None, random_state=None):
    #     """
    #     Approximate the distribution of the sum of random variable with a
    #     given dependence structure.
    #     The distribution can be accessed via the ``dist`` property, which is a ``distributions.PWC`` object.
        
    #     :param size: random variates sample size.
    #     :type size: ``int``
    #     :param random_state: random state for the random number generator.
    #     :type random_state: ``int``
    #     :return: Void.
    #     :rtype: ``None``
    #     """

    #     if (size is None) and (self.size is None):
    #         logger.warning('Distribution calculation is omitted as size is missing')
    #         return

    #     if size is not None:
    #         self.size = size
    #     if random_state is not None:
    #         self.random_state = random_state

    #     xsim = self._mc_rvs(
    #         self.size,
    #         self.random_state
    #     )
    #     # sorted and unique values
    #     x_ = np.unique(xsim)
    #     cdf_ = hf.ecdf(xsim)(x_)

    #     self.__dist = distributions.PWC(
    #         nodes=x_,
    #         cumprobs=cdf_
    #     )
    #     return

    # def _mc_rvs(self, size, random_state):
    #     """
    #     Monte Carlo random variates generator function of the sum of positive random variables.

    #     :param size: random variates sample size.
    #     :type size: ``int``
    #     :param random_state: random state for the random number generator.
    #     :type random_state: ``int``
    #     :return: sample of the sum of positive random variables.
    #     :rtype: ``numpy.ndarray``
    #     """
    #     u_ = self._copula_rvs(size, random_state).T
    #     return np.sum(self._margins_ppf(u_), axis=0)

    # def _private_prop_aep_initiate(self, x):
    #     """
    #     AEP algorithm helper function.
    #     See Arbenz P., Embrechts P., and Puccetti G.
    #     "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

    #     :param x: initial value for the quantile where the cumulative distribution function is evaluated.
    #     :type x: ``float``
    #     :return: Void
    #     :rtype: ``None``
    #     """
    #     self.__b = np.repeat(0, self.d).reshape(1, self.d)  # Vector b of the AEP algorithm.
    #     self.__h = np.array([[x]])  # Vector h of the AEP algorithm.
    #     self.__sn = np.array([1])  # Array of +1,-1, 0 indicating whether a volume must be summed,
    #     # subtracted or ignored, respectively.
    #     self.__vols = 0  # sum of 'volumes' * 'sn' used in AEP iteration

    # def _private_prop_aep_delete(self):
    #     """
    #     AEP algorithm helper function.
    #     See Arbenz P., Embrechts P., and Puccetti G.
    #     "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        
    #     :return: void
    #     :rtype: ``None``
    #     """
    #     del self.__b
    #     del self.__h
    #     del self.__sn
    #     del self.__vols

    # def _copula_rvs(self, size, random_state):
    #     """
    #     Copula random variates generator function.

    #     :param size: random variates sample size.
    #     :type size: ``int``
    #     :param random_state: random state for the random number generator.
    #     :type random_state: ``int``
    #     :return: random variates.
    #     :rtype: ``numpy.int`` or ``numpy.ndarray``
    #     """
    #     result = eval(config.COP_DICT[self.copula])(**self.copula_par).rvs(size, random_state)
    #     return np.array(result)

    # def _copula_cdf(self, k):
    #     """
    #     Copula cumulative distribution function.

    #     :param x: quantiles where the cumulative distribution function is evaluated.
    #     :type x: ``float`` or ``int`` or ``numpy.ndarray``
    #     :return: cumulative distribution function.
    #     :rtype: ``numpy.float64`` or ``numpy.ndarray``
    #     """
    #     result = eval(config.COP_DICT[self.copula])(**self.copula_par).cdf(k.transpose())
    #     return np.array(result)

    # def _margins_ppf(self, k):
    #     """
    #     Margin percent point function, a.k.a. the quantile function,
    #     inverse of the cumulative distribution function.

    #     :param k: probability.
    #     :type k: ``float`` or ``numpy.ndarray``
    #     :return: quantile.
    #     :rtype: ``numpy.float64`` or ``numpy.ndarray``
    #     """
    #     result = [eval(config.DIST_DICT[self.margins[j]])(**self.margins_pars[j]).ppf(k[j, :]) for j in range(self.d)]
    #     return np.array(result)

    # def _margins_cdf(self, k):
    #     """
    #     Margin cumulative distribution function.

    #     :param k: quantiles where the cumulative distribution function is evaluated.
    #     :type k: ``float`` or ``int`` or ``numpy.ndarray``
    #     :return: cumulative distribution function.
    #     :rtype: ``numpy.float64`` or ``numpy.ndarray``
    #     """
    #     result = [eval(config.DIST_DICT[self.margins[j]])(**self.margins_pars[j]).cdf(k[j, :]) for j in range(self.d)]
    #     return np.array(result)

    # def _volume_calc(self):
    #     """
    #     AEP algorithm helper function.
    #     See Arbenz P., Embrechts P., and Puccetti G.
    #     "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
    #     """
    #     mat_ = np.expand_dims(self._mat, axis=2)
    #     h_ = self._a * self.__h
    #     b_ = np.expand_dims(self.__b.T, axis=0)
    #     s_ = self._s.reshape(-1, 1)
    #     v_ = np.hstack((b_ + h_ * mat_))  # np.dstack(zip( (b_ + h_*mat_) ))[0]    # <- WARNING
    #     c_ = self._copula_cdf(self._margins_cdf(v_)).reshape(-1, self.__b.shape[0])
    #     result = np.sum(c_ * (s_ * np.sign(h_) ** self.d), axis=0)
    #     return result

    # def _sn_update(self):
    #     """
    #     AEP algorithm helper function.
    #     See Arbenz P., Embrechts P., and Puccetti G.
    #     "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
    #     """
    #     result = np.repeat(self.__sn, self._n_simpleces) * np.tile(self._m, self.__sn.shape[0])
    #     return result

    # def _h_update(self):
    #     """
    #     AEP algorithm helper function.
    #     See Arbenz P., Embrechts P., and Puccetti G.
    #     "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
    #     """
    #     result = (1 - np.tile(self._card, len(self.__h)) * self._a) * np.repeat(self.__h, len(self._card))
    #     return result

    # def _b_update(self):
    #     """
    #     AEP algorithm helper function.
    #     See Arbenz P., Embrechts P., and Puccetti G.
    #     "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
    #     """
    #     mat_ = self._mat[1:, :].transpose()
    #     h_ = np.repeat(self.__h, self._n_simpleces).reshape(-1, 1)
    #     times_ = int(h_.shape[0] / mat_.shape[1])
    #     result = np.repeat(self.__b, self._n_simpleces, 0)
    #     result = result + self._a * np.tile(h_, (1, self.d)) * np.tile(mat_, times_).transpose()
    #     return result

    # def _aep_cdf(self, x, n_iter):
    #     """
    #     AEP algorithm to approximate cdf.
    #     See Arbenz P., Embrechts P., and Puccetti G.
    #     "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

    #     :param x: quantile where the cumulative distribution function is evaluated.
    #     :type x: ``float``
    #     :param n_iter: number of algorithm iterations.
    #     :type n_iter: ``int``

    #     :return: cumulative distribution function.
    #     :rtype: ``numpy.float64`` or ``numpy.ndarray``
    #     """
    #     self._private_prop_aep_initiate(x)
    #     cdf = self._volume_calc()[0]
    #     for _ in range(n_iter):
    #         self.__sn = self._sn_update()
    #         self.__b = self._b_update()
    #         self.__h = self._h_update()
    #         self.__vols = np.sum(self.__sn * self._volume_calc())
    #         cdf += self.__vols
    #     cdf += self.__vols * (self._ext - 1)
    #     self._private_prop_aep_delete()
    #     return cdf

    # def _mc_cdf(self, x):
    #     """
    #     Cumulative distribution function from Monte Carlo simulation.

    #     :param x: quantile where the cumulative distribution function is evaluated.
    #     :type x: ``int`` or ``float``

    #     :return: cumulative distribution function.
    #     :rtype: ``numpy.float64`` or ``numpy.ndarray``
    #     """
    #     if not self._check_missing_dist():
    #         return self.dist.cdf(x)

    # def _mc_ppf(self, q):
    #     """
    #     Monte Carlo percent point function, a.k.a. the quantile function, of the random variable sum.
    #     Inverse of cumulative distribution function.
        
    #     :param q: level at which the percent point function is evaluated.
    #     :type q: ``float``, ``numpy.ndarray``, ``numpy.floating``
        
    #     :return: percent point function.
    #     :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
    #     """
    #     if not self._check_missing_dist():
    #         return self.dist.ppf(q)

    def cdf(self, x, method='mc', n_iter=7):
        """
        Cumulative distribution function of the random variable sum.
        If ``method`` is Monte Carlo ('mc') the function relies on the approximated distribution
        calculated when the LossAggregation class is initiated (accessed via the ``dist`` property of
        LossAggregation).
        If ``method`` is AEP ('aep') the cdf is evaluated on-the-fly regardless of the ``dist`` property).
        
        :param x: quantile where the cumulative distribution function is evaluated.
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
        isscalar = not isinstance(x, (np.ndarray, list)) 
        x = np.ravel(x)

        output = None   
        if method == 'aep':
            hf.assert_type_value(
                n_iter, 'n_iter', logger, (int, float),
                lower_bound=1, lower_close=True
            )
            n_iter = int(n_iter)
            output = np.empty(len(x))
            for i in range(len(output)):
                output[i] = self.calc.aep.cdf(x[i], n_iter)
            if isscalar:
                output = output.item()
        else:
            if not self._check_missing_dist():
                output = self.calc.mc.cdf(x, self.dist)

        return output

    def sf(self, x, method='mc', n_iter=None):
        """
        Survival function of the random variable sum.
        If ``method`` is Monte Carlo ('mc') the function relies on the approximated distribution
        calculated when the LossAggregation class is initiated (accessed via the ``dist`` property).
        If ``method`` is AEP ('aep') the survival function is evaluated pointwise on-the-fly regardless of the ``dist`` property.
        
        :param x: quantile where the survival function is evaluated.
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

    def ppf(self, q, method='mc', n_iter=7, tolerance=1e-5, max_search_iter=10):
        """
        Percent point function, a.k.a. the quantile function, of the random variable sum.
        Inverse of cumulative distribution function.
        
        :param q: level at which the percent point function is evaluated.
        :type q: ``float``, ``numpy.ndarray``, ``numpy.floating``
        :param method: method to approximate the cdf of the sum of the random variables.
                        One of AEP ('aep') and Monte Carlo simulation ('mc').
        :type method: ``string``
        :param n_iter: number of AEP algorithm iterations (optional).
        :type n_iter: ``int``
        :param tolerance: tolerance for searching algorithm iterations to approximate the ppf via AEP method (optional).
        :type tolerance: ``float``
        :param max_search_iter: maximum number of searching algorithm iterations to approximate the ppf via AEP method (optional).
        :type max_search_iter: ``int``

        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        hf.assert_member(
            method,
            config.LOSS_AGGREGATION_METHOD,
            logger
        )
        
        if method == "aep":
            return self.calc.aep.ppf(q, n_iter, tolerance, max_search_iter, self.dist)
        else:
            return self.calc.mc.ppf(q)

    # def _aep_ppf(self, q, n_iter, tolerance, max_search_iter):
    #     """
    #     Percent point function, a.k.a. the quantile function, of the random variable sum
    #     using AEP algorithm.
    #     Inverse of cumulative distribution function.
        
    #     :param q: level at which the percent point function is evaluated.
    #     :type q: ``float``, ``numpy.ndarray``, ``numpy.floating``
    #     :param n_iter: number of AEP algorithm iterations (optional).
    #     :type n_iter: ``int``
    #     :param tolerance: tolerance threshold, maximum allowed difference between cumulative probability values.
    #     :type tolerance: ``float``
    #     :param max_search_iter: maximum number of searching algorithm iterations to approximate the ppf.
    #     :type max_search_iter: ``int``

    #     :return: percent point function.
    #     :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
    #     """
        
    #     hf.assert_type_value(
    #         q, 'q', logger, (np.floating, int, float, list, np.ndarray)
    #         )
    #     isscalar = not isinstance(q, (np.ndarray, list)) 
    #     q = np.ravel(q)
    #     if np.any(q > 1):
    #         message = 'Make sure q is lower than or equal to 1'
    #         logger.error(message)
    #         raise ValueError(message)
    #     if np.any(q < 0):
    #         message = 'Make sure q is higher than or equal to 0'
    #         logger.error(message)
    #         raise ValueError(message)

    #     output = np.empty(q.shape)
    #     for idx in range(q.shape[0]):

    #         if q[idx] >= self.dist.cumprobs[-1]:
    #             return self.dist.max
    #         elif q[idx] <= self.dist.cumprobs[0]:
    #             return self.dist.min

    #         idx_right = np.searchsorted(self.dist.cumprobs, q[idx], side='left')
    #         idx_left = np.max(idx_right - 1, 0)
    #         count = 0

    #         output[idx] = self._aep_binary_search(
    #             q[idx],
    #             self.dist.nodes[idx_left],  self.dist.nodes[idx_right],
    #             self.dist.cumprobs[idx_left], self.dist.cumprobs[idx_right],
    #             n_iter,
    #             tolerance,
    #             max_search_iter,
    #             count
    #             )
    #         idx += 1
        
    #     output = output.item() if isscalar else output
    #     return output
        
    # def _aep_binary_search(
    #         self,
    #         q,
    #         x_left, x_right,
    #         q_left, q_right,
    #         n_iter,
    #         tolerance,
    #         max_search_iter,
    #         count
    #         ):
    #     """
    #     Binary search searching algorithm for approximating the ppf via AEP method.
    #     Recursive function.
        
    #     :param q: level at which the percent point function is evaluated.
    #     :type q: ``float``, ``numpy.floating``
    #     :param q_left: largest level neighbour smaller than q.
    #     :type q_left: ``float``, ``numpy.floating``
    #     :param q_right: smallest level neighbour larger than q.
    #     :type q_right: ``float``, ``numpy.floating``
    #     :param x_left:  quantile associated to q_left level.
    #     :type x_left: ``float``, ``numpy.floating``
    #     :param x_right: quantile associated to q_right level.
    #     :type x_right: ``float``, ``numpy.floating``
    #     :param method: method to approximate the cdf of the sum of the random variables.
    #                     One of AEP ('aep') and Monte Carlo simulation ('mc').
    #     :type method: ``string``
    #     :param n_iter: number of AEP algorithm iterations (optional).
    #     :type n_iter: ``int``
    #     :param tolerance: tolerance threshold, maximum allowed difference between cumulative probability values.
    #     :type tolerance: ``float``
    #     :param max_search_iter: maximum number of searching algorithm iterations to approximate the ppf.
    #     :type max_search_iter: ``int``
    #     :param count: counter of the current searching algorithm iteration.
    #     :type count: ``int``

    #     :return: percent point function.
    #     :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
    #     """
        
    #     if abs(q_left - q) <= abs(q_right - q):
    #         if abs(q_left - q) <= tolerance:
    #             return x_left
    #     else:
    #         if abs(q_right - q) <= tolerance:
    #             return x_right
        
    #     x_mid = (x_left + x_right) / 2
    #     q_mid = self._aep_cdf(x_mid, n_iter)

    #     if count > max_search_iter:
    #         return q_mid

    #     if q_mid > q:
    #         return self._aep_binary_search(
    #             q, x_left, x_mid, q_left, q_mid,
    #             tolerance, n_iter,
    #             max_search_iter, count+1
    #             )
    #     else:
    #         return self._aep_binary_search(
    #             q, x_mid, x_right, q_mid, q_right,
    #             tolerance, n_iter,
    #             max_search_iter, count+1
    #             )

    def moment(self, central=False, n=1):
        """
        Moment of order n of the random variable sum.
        Piecewise-constant approximation.

        :param central: ``True`` if the moment is central, ``False`` if the moment is raw.
        :type central: ``bool``
        :param n: order of the moment, optional (default is 1).
        :type n: ``int``
        :return: moment of order n.
        :rtype: ``numpy.float64``
        """
        if not self._check_missing_dist():
            return self.dist.moment(central, n)

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        if not self._check_missing_dist():
            return self.dist.rvs(size, random_state)

    def mean(self):
        """
        Mean of the random variable sum.
        Piecewise-constant approximation.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        if not self._check_missing_dist():
            return self.dist.mean()
    
    def skewness(self):
        """
        Skewness of the random variable sum.
        Piecewise-constant approximation.

        :return: skewness.
        :rtype: ``numpy.float64``
        """
        if not self._check_missing_dist():
            return self.dist.skewness()

    def std(self):
        """
        Standard deviation of the random variable sum.
        Piecewise-constant approximation.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        if not self._check_missing_dist():
            return self.dist.std()

    def _check_missing_dist(self):
        """
        Check whether the distribution of the random variable sum is missing.
        Helper method called before executing other methods based on ``dist`` property.

        :return: Check outcome
        :rtype: ``bool``
        """
        if self.dist is None:
            logger.warning('Execution is stopped as dist is missing. Plese run dist_calculate method first.')
            return True
        else:
            return False

    def plot_cdf(self, log_x_scale=False, log_y_scale=False, **kwargs):
        """
        Plot the cumulative distribution function of the random variable sum.

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
