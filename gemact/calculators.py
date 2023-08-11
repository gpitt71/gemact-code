from .libraries import *
from . import helperfunctions as hf
from . import config
from .distributions import PWC 

quick_setup()
logger = log.name('calculators')

class LossModelCalculator:
    """
    Calculation methods used in LossModel and Severity classes. 
    Python informal static class.
    """

    def __init__():
        pass

    @staticmethod
    def fast_fourier_transform(severity, frequency, n_aggr_dist_nodes, discr_step, tilt, tilt_value, normalize=False):
        """
        Aggregate loss distribution via Fast Fourier Transform.

        :param severity: discretized severity, nodes sequence and discrete probabilities.
        :type severity: ``dict``
        :param frequency: frequency model (adjusted).
        :type frequency: ``Frequency``
        :param n_aggr_dist_nodes: number of nodes in the approximated aggregate loss distribution.
        :type n_aggr_dist_nodes: ``int``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :param tilt_value: tilting parameter value of FFT method for the aggregate loss distribution approximation.
        :type tilt_value: ``float``
        :param tilt: whether tilting of FFT is present or not.
        :type tilt: ``bool``
        :return: aggregate loss distribution empirical pmf, cdf, nodes
        :rtype: ``dict``
        """
        
        fj = severity['fj']

        if tilt:
            tilting_par = 20 / n_aggr_dist_nodes if tilt_value is None else tilt_value
        else:
            tilting_par = 0

        fj = np.append(fj, np.repeat(0, n_aggr_dist_nodes - fj.shape[0]))
        
        f_hat = fft(np.exp(-tilting_par * np.arange(0, n_aggr_dist_nodes, step=1)) * fj)
        g_hat = frequency.model.pgf(f=f_hat)
        g = np.exp(tilting_par * np.arange(0, n_aggr_dist_nodes, step=1)) * np.real(ifft(g_hat))

        if normalize:
            g = g / np.sum(g)

        cum_probs = np.minimum(np.cumsum(g), 1)
        
        if (1 - cum_probs[-1]) > config.PROB_TOLERANCE:
            message = 'Failure to obtain a cumulative distribution function close to 1. '\
                'Last calculated cumulative probability is %s.' % ("{:.4f}".format(cum_probs[-1]))
            logger.warning(message)

        return {'cdf': cum_probs,
                'nodes': discr_step * np.arange(0, n_aggr_dist_nodes, step=1)}

    @staticmethod
    def panjer_recursion(frequency, severity, n_aggr_dist_nodes, discr_step, normalize=False):
        """
        Aggregate loss distribution via Panjer recursion.

        :param severity: discretized severity, nodes sequence and discrete probabilities.
        :type severity: ``dict``
        :param frequency: frequency model (adjusted).
        :type frequency: ``Frequency``
        :param n_aggr_dist_nodes: number of nodes in the approximated aggregate loss distribution.
        :type n_aggr_dist_nodes: ``int``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :return: aggregate loss distribution empirical pdf, cdf, nodes
        :rtype: ``dict``
        """
        
        fj = severity['fj']
        a, b, p0, g = frequency.abp0g0(fj)

        fj = np.append(fj, np.repeat(0, n_aggr_dist_nodes - fj.shape[0]))
        # z_ = np.arange(1, n_aggr_dist_nodes + 1)
        fpmf = frequency.model.pmf(1)
        for j in range(1, n_aggr_dist_nodes):
            g = np.insert(g,
            0, # position
            (np.sum(
                ((a + b * np.arange(1, j + 1) / j) * fj[1:(j+1)] * g[:j]))
                )
            )
        g = ((fpmf - (a + b) * p0) * fj + g[::-1]) / (1 - a * fj[0])
        
        if normalize:
            g = g / np.sum(g)
        
        cum_probs = np.minimum(np.cumsum(g), 1)
        
        if (1 - cum_probs[-1]) > config.PROB_TOLERANCE:
            message = 'Failure to obtain a cumulative distribution function close to 1. '\
                'Last calculated cumulative probability is %s.' % ("{:.4f}".format(cum_probs[-1]))
            logger.warning(message)

        return {'cdf': cum_probs,
                'nodes': discr_step * np.arange(0, n_aggr_dist_nodes, step=1)}

    @staticmethod
    def mc_simulation(severity, frequency, cover, deductible, n_sim, random_state):
        """
        Aggregate loss distribution via Monte Carlo simulation.

        :param severity: severity model.
        :type severity: ``Severity``
        :param frequency: frequency model (adjusted).
        :type frequency: ``Frequency``
        :param cover: cover, also referred to as limit.
        :type cover: ``int`` or ``float``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int`` or ``float``
        :param n_sim: number of simulations.
        :type n_sim: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``
        :return: aggregate loss distribution empirical pdf, cdf, nodes.
        :rtype: ``dict``
        """
                
        p0 = severity.model.cdf(deductible) if deductible > 1e-05 else 0.

        fqsample = frequency.model.rvs(n_sim, random_state=random_state)        
        np.random.seed(random_state+1)
        svsample = severity.model.ppf(
            np.random.uniform(low=p0, high=1.0, size=int(np.sum(fqsample)))
        )
        svsample = np.minimum(svsample - deductible, cover)
        # cumsum excluding last entry as not needed in subsequent row calculation
        cs = np.cumsum(fqsample).astype(int)[:(n_sim-1)]
        xsim = np.stack([*map(np.sum, np.split(svsample, cs))])

        x_ = np.unique(xsim)
        cdf_ = hf.ecdf(xsim)(x_)

        return {'cdf': cdf_,
                'nodes': x_}

    @staticmethod
    def mass_dispersal(severity, deductible, exit_point, discr_step, n_discr_nodes):
        """
        Severity discretization according to the mass dispersal method.

        :param severity: severity model.
        :type severity: ``Severity``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int`` or ``float``
        :param exit_point: severity 'exit point', deductible plus cover.
        :type cover: ``int`` or ``float``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :param n_discr_nodes: number of nodes of the discretized severity.
        :type n_discr_nodes: ``int``
        :return: discrete severity, nodes sequence and discrete probabilities.
        :rtype: ``dict``
        """
        f0 = (severity.model.cdf(deductible + discr_step / 2) - severity.model.cdf(deductible)) / \
             (1 - severity.model.cdf(deductible))
        nodes = np.arange(0, n_discr_nodes) + .5

        fj = np.append(
            f0,
            (severity.model.cdf(deductible + nodes * discr_step)[1:] -
            severity.model.cdf(deductible + nodes * discr_step)[:-1]) /
            (1 - severity.model.cdf(deductible))
        )

        if exit_point != float('inf'):
            fj = np.append(fj, (1 - severity.model.cdf(
                exit_point - discr_step / 2)) / (
                    1 - severity.model.cdf(deductible)))

        nodes = severity.loc + np.arange(0, n_discr_nodes) * discr_step

        if exit_point != float('inf'):
            nodes = np.concatenate((nodes, [nodes[-1] + discr_step]))


        return {'nodes': nodes, 'fj': fj}

    @staticmethod
    def lower_discretization(severity, deductible, exit_point, discr_step, n_discr_nodes):
        """
        Severity discretization according to the lower discretization method.

        :param severity: severity model.
        :type severity: ``Severity``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int`` or ``float``
        :param exit_point: severity 'exit point', deductible plus cover.
        :type cover: ``int`` or ``float``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :param n_discr_nodes: number of nodes of the discretized severity.
        :type n_discr_nodes: ``int``
        :return: discrete severity, nodes sequence and discrete probabilities.
        :rtype: ``dict``
        """
        f0 =(severity.model.cdf(deductible)-severity.model.cdf(deductible)) / \
             (1 - severity.model.cdf(deductible))
        nodes = np.arange(0, n_discr_nodes)

        fj = np.append(
            f0,
            (severity.model.cdf(deductible + nodes * discr_step)[1:] -
             severity.model.cdf(deductible + nodes * discr_step)[:-1]) /
            (1 - severity.model.cdf(deductible))
        )


        if exit_point != float('inf'):
            fj = np.append(fj, (1 - severity.model.cdf(
                exit_point - discr_step)) / (
                                   1 - severity.model.cdf(deductible)))

        nodes = severity.loc + np.arange(0, n_discr_nodes) * discr_step

        if exit_point != float('inf'):
            nodes = np.concatenate((nodes, [nodes[-1] + discr_step]))

        return {'nodes': nodes, 'fj': fj}

    @staticmethod
    def upper_discretization(severity, deductible, exit_point, discr_step, n_discr_nodes):
        """
        Severity discretization according to the upper discretization method.

        :param severity: severity model.
        :type severity: ``Severity``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int`` or ``float``
        :param exit_point: severity 'exit point', deductible plus cover.
        :type cover: ``int`` or ``float``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :param n_discr_nodes: number of nodes of the discretized severity.
        :type n_discr_nodes: ``int``
        :return: discrete severity, nodes sequence and discrete probabilities.
        :rtype: ``dict``
        """
        # extramass = (severity.model.cdf(deductible)) / \
        #      (1 - severity.model.cdf(deductible))
        nodes = np.arange(0, n_discr_nodes+1)

        fj = (severity.model.cdf(deductible + nodes * discr_step)[1:] - severity.model.cdf(deductible + nodes * discr_step)[:-1]) /(1 - severity.model.cdf(deductible))


        if exit_point != float('inf'):
            fj = np.append(fj, (1 - severity.model.cdf(
                exit_point)) / (
                                   1 - severity.model.cdf(deductible)))

        nodes = severity.loc + np.arange(0, n_discr_nodes) * discr_step


        return {'nodes': nodes, 'fj': fj}

    @staticmethod
    def upper_discr_point_prob_adjuster(severity, deductible, exit_point, discr_step):
        """
        Probability of the discretization upper point in the local moment.
        In case an upper priority on the severity is provided, the probability of the node sequence upper point
        is adjusted to be coherent with discretization step size and number of nodes.

        :param severity: severity model.
        :type severity: ``Severity``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int`` or ``float``
        :param exit_point: severity 'exit point', deductible plus cover.
        :type cover: ``int`` or ``float``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :return: probability mass in (u-d/h)*m
        :rtype: ``numpy.ndarray``
        """

        if exit_point == float('inf'):
            output = np.array([])
        else:
            output = (severity.model.lev(exit_point - severity.loc) -
                severity.model.lev(exit_point - severity.loc - discr_step)) / \
                    (discr_step * severity.model.den(low=deductible, loc=severity.loc))
        return output

    @staticmethod
    def local_moments(severity, deductible, exit_point, discr_step, n_discr_nodes):
        """
        Severity discretization according to the local moments method.

        :param severity: severity model.
        :type severity: ``Severity``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int`` or ``float``
        :param exit_point: severity 'exit point', deductible plus cover.
        :type cover: ``int`` or ``float``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :param n_discr_nodes: number of nodes of the discretized severity.
        :type n_discr_nodes: ``int``
        :return: discrete severity, nodes sequence and discrete probabilities.
        :rtype: ``dict``
        """

        last_node_prob = LossModelCalculator.upper_discr_point_prob_adjuster(
            severity, deductible, exit_point, discr_step
            )

        n = severity.model.lev(
            deductible + discr_step - severity.loc
            ) - severity.model.lev(
                deductible - severity.loc
                )

        den = discr_step * severity.model.den(low=deductible, loc=severity.loc)
        nj = 2 * severity.model.lev(deductible - severity.loc + np.arange(
            1, n_discr_nodes) * discr_step) - severity.model.lev(
            deductible - severity.loc + np.arange(
                0, n_discr_nodes - 1) * discr_step) - severity.model.lev(
            deductible - severity.loc + np.arange(2, n_discr_nodes + 1) * discr_step)

        fj = np.append(1 - n / den, nj / den)

        nodes = severity.loc + np.arange(0, n_discr_nodes) * discr_step
        if exit_point != float('inf'):
            nodes = np.concatenate((nodes, [nodes[-1] + discr_step]))
        return {'nodes': nodes, 'fj': np.append(fj, last_node_prob)}

class LossAggregationCalculator:
    """
    Calculation class used in LossAggregation.
    """
    def __init__(
        self,
        copula,
        margins
        ):
        hf.check_condition(
            margins.dim, copula.dim, 'number of margins', logger, '=='
        )
        self.aep = AEPCalculator(
            copula=copula,
            margins=margins
        )
        self.mc = MCCalculator(
            copula=copula,
            margins=margins
        )

    @property
    def mc(self):
        return self.__mc
    
    @mc.setter
    def mc(self, value):
        hf.assert_type_value(
            value, 'mc', logger, MCCalculator
        )
        self.__mc = value

    @property
    def aep(self):
        return self.__aep
    
    @aep.setter
    def aep(self, value):
        hf.assert_type_value(
            value, 'aep', logger, AEPCalculator
        )
        self.__aep = value

class AEPCalculator:
    """
    Calculation class part of LossAggregationCalculator used in LossAggregation.
    Class representing the AEP algorithm calculator.

    :param copula: Copula (dependence structure between margins).
    :type copula: ``Copula``
    :param margins: Marginal distributions.
    :type margins: ``Margins``
    """

    def __init__(
        self,
        copula,
        # copula_par,
        margins,
        # margins_pars
        ):
        self.copula = copula
        # self.copula_par = copula_par
        self.margins = margins
        # self.margins_pars = margins_pars
        hf.check_condition(
            copula.dim, config.DCEILING, 'copula dimension', logger, '<='
        )
        hf.check_condition(
            margins.dim, copula.dim, 'number of margins', logger, '=='
        )

    # @property
    # def margins_pars(self):
    #     return self.__margins_pars

    # @margins_pars.setter
    # def margins_pars(self, value):
        # hf.assert_type_value(value, 'margins_pars', logger, type=(list))
        # hf.check_condition(
        #     len(value), len(self.margins), 'margins_pars', logger
        # )
        
        # for j in range(len(value)):
        #     hf.assert_type_value(value[j], 'margins_pars item', logger, type=(dict))
            
        #     try:
        #         eval(config.DIST_DICT[self.margins[j]])(**value[j])
        #     except Exception:
        #         logger.error('Please make sure that margin %s is correctly parametrized.\n See %s' % (j+1, config.SITE_LINK))
        #         raise
        # self.__margins_pars = value

    @property
    def margins(self):
        return self.__margins

    @margins.setter
    def margins(self, value):
        # hf.assert_type_value(value, 'margins', logger, type=(list))
        # hf.assert_type_value(len(value), 'margins length', logger, type=(float, int),
        #     upper_bound=config.DCEILING)

        # for j in range(len(value)):
        #     hf.assert_member(value[j], config.DIST_DICT, logger, config.SITE_LINK)
        #     hf.assert_member('severity', eval(config.DIST_DICT[value[j]]).category(), logger, config.SITE_LINK)
        self.__margins = value

    @property
    def copula(self):
        return self.__copula

    @copula.setter
    def copula(self, value):
        # hf.assert_type_value(value, 'copula', logger, type=(str))
        # hf.assert_member(value, config.COP_DICT, logger, config.SITE_LINK)
        self.__copula = value

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
    def d(self):
        return self.copula.dim

    @property
    def _a(self):
        # Alpha parameter of the AEP algorithm.
        return 2. / (self.d + 1)

    @property
    def _ext(self):
        # Probability correction of the AEP
        return ((self.d + 1) ** self.d) / (special.factorial(self.d) * 2 ** self.d)

    @property
    def _mat(self):
        # Matrix of the vectors in the {0,1}**d space.
        return hf.cartesian_product(*([np.array([0, 1])] * self.d)).T

    @property
    def _n_simpleces(self):
        # AEP number of new simpleces received in each step.
        return 2 ** self.d - 1

    @property
    def _card(self):
        # AEP cardinality of the 'mat' matrix.
        return np.sum(self._mat, axis=1)[1:]

    @property
    def _s(self):
        # AEP array of +1 or -1, indicating whether to sum or subtract a volume, respectively.
        return (-1) ** (self.d - np.sum(self._mat, axis=1))

    @property
    def _m(self):
        # Array of +1, -1, 0, indicating whether the new simpleces origined from sn must be summed,
        # subtracted or ignored, respectively.
        output = self._card.copy()
        greater = np.where(output > (1 / self._a))
        equal = np.where(output == (1 / self._a))
        lower = np.where(output < (1 / self._a))
        output[greater] = (-1) ** (self.d + 1 - output[greater])
        output[equal] = 0
        output[lower] = (-1) ** (1 + output[lower])
        return output

    @property
    def dist(self):
        return self.__dist

    def _private_prop_aep_initiate(self, x):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param x: initial value for the quantile where the cumulative distribution function is evaluated.
        :type x: ``float``
        :return: Void
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

    # def _copula_rvs(self, copula, size, random_state):
    #     """
    #     Copula random variates generator function.

    #     :param size: random variates sample size.
    #     :type size: ``int``
    #     :param random_state: random state for the random number generator.
    #     :type random_state: ``int``
    #     :return: random variates.
    #     :rtype: ``numpy.int`` or ``numpy.ndarray``
    #     """
    #     result = np.array(copula.rvs(size, random_state))
    #     # result = eval(config.COP_DICT[self.copula])(**self.copula_par).rvs(size, random_state)
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

    def _volume_calc(self):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        """
        mat_ = np.expand_dims(self._mat, axis=2)
        h_ = self._a * self.__h
        b_ = np.expand_dims(self.__b.T, axis=0)
        s_ = self._s.reshape(-1, 1)
        v_ = np.hstack((b_ + h_ * mat_))  # np.dstack(zip( (b_ + h_*mat_) ))[0]    # <- WARNING
        c_ = self.copula.cdf(self.margins.cdf(v_)).reshape(-1, self.__b.shape[0])
        result = np.sum(c_ * (s_ * np.sign(h_) ** self.d), axis=0)
        return result

    def _sn_update(self):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        """
        result = np.repeat(self.__sn, self._n_simpleces) * np.tile(self._m, self.__sn.shape[0])
        return result

    def _h_update(self):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        """
        result = (1 - np.tile(self._card, len(self.__h)) * self._a) * np.repeat(self.__h, len(self._card))
        return result

    def _b_update(self):
        """
        AEP algorithm helper function.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        """
        mat_ = self._mat[1:, :].transpose()
        h_ = np.repeat(self.__h, self._n_simpleces).reshape(-1, 1)
        times_ = int(h_.shape[0] / mat_.shape[1])
        result = np.repeat(self.__b, self._n_simpleces, 0)
        result = result + self._a * np.tile(h_, (1, self.d)) * np.tile(mat_, times_).transpose()
        return result

    def cdf(self, x, n_iter):
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
        cdf += self.__vols * (self._ext - 1)
        self._private_prop_aep_delete()
        return cdf

    def ppf(self, q, n_iter, tol=1e-04):
        """
        Percent point function, a.k.a. the quantile function, of the random variable sum
        using AEP algorithm. 
        Inverse of cumulative distribution function. See ``scipy.optimize.brentq``.
        
        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :param n_iter: number of AEP algorithm iterations (optional).
        :type n_iter: ``int``
        :param tol: tolerance threshold, maximum allowed absolute difference between cumulative probability values (optional).
        :type tol: ``float``

        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """

        q = np.ravel(q)
        qarr = np.repeat(q, self.d).reshape(self.d, -1)
        x0 = np.sum(self.margins.ppf(qarr))
        diff = (q - self.cdf(x0, n_iter=n_iter))

        if abs(diff) <= tol:
            return x0
        
        if diff < 0:
            # x0 is larger than actual q-th level quantile
            x1 = 0.5 * x0
            while self.cdf(x1, n_iter=n_iter) > q:
                # substitute x0 with x1 as it narrows the interval
                x0 = x1
                x1 = 0.5 * x1
            bracket = [x1, x0]
        else: # diff > 0:
            # x0 is smaller than actual q-th level quantile
            x1 = 2 * x0
            while self.cdf(x1, n_iter=n_iter) < q:
                # substitute x0 with x1 as it narrows the interval
                x0 = x1
                x1 = 2 * x1
            bracket = [x0, x1]

        output = root_scalar(lambda x : q - self.cdf(x, n_iter=n_iter), method='brentq', bracket=bracket, xtol=tol)
        if output.converged:
            return output.root
        else:
            logger.warning('Execution of ppf failed. Result does not converged')
            return np.nan

    # def ppf(self, q, n_iter=7, tolerance=1e-5, max_search_iter=10, dist=None):
    #     """
    #     Percent point function, a.k.a. the quantile function, of the random variable sum
    #     using AEP algorithm.
    #     Inverse of cumulative distribution function.
        
    #     :param q: level at which the percent point function is evaluated.
    #     :type q: ``float``, ``numpy.ndarray``, ``numpy.floating``
    #     :param n_iter: number of AEP algorithm iterations (optional).
    #     :type n_iter: ``int``
    #     :param tolerance: tolerance threshold, maximum allowed difference between cumulative probability values (optional).
    #     :type tolerance: ``float``
    #     :param max_search_iter: maximum number of searching algorithm iterations to approximate the ppf (optional).
    #     :type max_search_iter: ``int``
    #     :param dist: Piecewise-constant distribution. It steers the searching algorithm.
    #     :type dist: ``PWC.Distribution``

    #     :return: percent point function.
    #     :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
    #     """
    #     if dist is None:
    #         logger.warning('Missing dist. AEP ppf cannot be calculated.')
    #         return

    #     hf.assert_type_value(
    #         dist, 'dist', logger, PWC
    #         )        
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

    #         if q[idx] >= dist.cumprobs[-1]:
    #             return dist.max
    #         elif q[idx] <= dist.cumprobs[0]:
    #             return dist.min

    #         idx_right = np.searchsorted(dist.cumprobs, q[idx], side='left')
    #         idx_left = np.max(idx_right - 1, 0)
    #         count = 0

    #         output[idx] = self.binary_search(
    #             q[idx],
    #             dist.nodes[idx_left],  dist.nodes[idx_right],
    #             dist.cumprobs[idx_left], dist.cumprobs[idx_right],
    #             n_iter,
    #             tolerance,
    #             max_search_iter,
    #             count
    #             )
    #         idx += 1
        
    #     output = output.item() if isscalar else output
    #     return output
        
    # def binary_search(
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
    #     q_mid = self.cdf(x_mid, n_iter)

    #     if count > max_search_iter:
    #         return q_mid

    #     if q_mid > q:
    #         return self.binary_search(
    #             q, x_left, x_mid, q_left, q_mid,
    #             tolerance, n_iter,
    #             max_search_iter, count+1
    #             )
    #     else:
    #         return self.binary_search(
    #             q, x_mid, x_right, q_mid, q_right,
    #             tolerance, n_iter,
    #             max_search_iter, count+1
    #             )

class MCCalculator:
    """
    Class representing the MC calculator.

    :param copula: name of the copula that describes the dependence structure.
    :type copula: ``str``
    :param margins: list of the marginal distributions.
    :type margins: ``list``
    """

    def __init__(
        self,
        copula,
        margins,
        # size,
        # random_state
        ):
        self.copula = copula
        self.margins = margins
        # self.random_state = random_state
        # self.size = size
        hf.check_condition(
            margins.dim, copula.dim, 'number of margins', logger, '=='
        )
    
    @property
    def margins(self):
        return self.__margins

    @margins.setter
    def margins(self, value):
        # hf.assert_type_value(value, 'margins', logger, type=(list))
        # hf.assert_type_value(len(value), 'margins length', logger, type=(float, int), upper_bound=config.DCEILING)
        self.__margins = value

    @property
    def copula(self):
        return self.__copula

    @copula.setter
    def copula(self, value):
        # hf.assert_type_value(value, 'copula', logger, type=(str))
        # hf.assert_member(value, config.COP_DICT, logger, config.SITE_LINK)
        self.__copula = value

    def rvs(self, size, random_state):
        """
        Monte Carlo random variates generator function of the sum of positive random variables.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``
        :return: sample of the sum of positive random variables.
        :rtype: ``numpy.ndarray``
        """
        u_ = self.copula.rvs(size, random_state).T
        return np.sum(self.margins.ppf(u_), axis=0)

    def cdf(self, x, dist):
        """
        Cumulative distribution function from Monte Carlo simulation.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int`` or ``float``

        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        hf.assert_type_value(
            dist, 'dist', logger, (PWC)
        )
        return dist.cdf(x)
    
    def ppf(self, q, dist):
        """
        Monte Carlo percent point function, a.k.a. the quantile function, of the random variable sum.
        Inverse of cumulative distribution function.
        
        :param q: level at which the percent point function is evaluated.
        :type q: ``float``, ``numpy.ndarray``, ``numpy.floating``
        
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        hf.assert_type_value(
            dist, 'dist', logger, (PWC)
        )
        return dist.ppf(q)
     
    def dist_calculate(self, size, random_state):
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

        # if (size is None) and (self.size is None):
        #     logger.warning('Distribution calculation is omitted as size is missing')
        #     return

        # if size is not None:
        #     self.size = size
        # if random_state is not None:
        #     self.random_state = random_state

        xsim = self.rvs(size, random_state)
        nodes = np.unique(xsim) # nodes: sorted and unique values
        cumprobs = hf.ecdf(xsim)(nodes)

        return (nodes, cumprobs)