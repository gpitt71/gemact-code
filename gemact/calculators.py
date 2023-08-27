from .libraries import *
from . import helperfunctions as hf
from . import config

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

        cum_probs = np.minimum(np.sort(np.cumsum(g)), 1) # avoid numerical issues float numbers
        
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
        
        cum_probs = np.minimum(np.sort(np.cumsum(g)), 1) # avoid numerical issues float numbers
        
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


class MCCalculator:
    """
    Class representing the Monte Carlo (MC) algorithm calculators.
    Calculation methods used in LossAggregation. 
    Python informal static class.
    """

    def __init__():
        pass
    
    @staticmethod
    def rvs(size, random_state, copula, margins):
        """
        Random variates generator function of the sum of positive random variables.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``
        :param copula: Copula (dependence structure between margins).
        :type copula: ``Copula``
        :param margins: Marginal distributions.
        :type margins: ``Margins``
        :return: sample of the sum of positive random variables.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        u_ = copula.rvs(size, random_state).T
        return np.sum(margins.ppf(u_), axis=0)

    @staticmethod
    def simulation_execute(size, random_state, copula, margins):
        """
        Execute Monte Carlo simulation to approximate the distribution of the sum of random variable with a
        given dependence structure.
        
        :param size: simulation random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``
        :param copula: Copula (dependence structure between margins).
        :type copula: ``Copula``
        :param margins: Marginal distributions.
        :type margins: ``Margins``
        :return: simulated nodes and their (empirical) cumulative probabilites.
        :rtype: ``tuple``
        """

        xsim = MCCalculator.rvs(size, random_state, copula, margins)
        nodes = np.unique(xsim) # nodes: sorted and unique values
        cumprobs = hf.ecdf(xsim)(nodes)

        return (nodes, cumprobs)


class AEPCalculator:
    """
    Class representing the AEP algorithm calculators.
    Calculation methods used in LossAggregation. 
    Python informal static class.
    """

    def __init__():
        pass

    @staticmethod
    def _mat(d):
        """
        AEP algorithm helper function.
        Generate matrix of the vectors in the {0,1}**d space.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param d: dimensionality of the space.
        :type d: ``int``
        :return: matrix of the vectors in the {0,1}**d space.
        :rtype: ``numpy.ndarray``
        """
        return hf.cartesian_product(*([np.array([0, 1])] * d)).T

    @staticmethod
    def _m(_card, d):
        """
        AEP algorithm helper function.
        Generate # Array of +1, -1, 0, indicating whether the new simpleces
        origined must be summed, subtracted or ignored, respectively.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param _card: cardinalities of matrices
        :type _card: ``numpy.ndarray``
        :param d: dimensionality of the space.
        :type d: ``int``
        :return: matrix of the vectors in the {0,1}**d space.
        :rtype: ``numpy.ndarray``
        """
        _a = (2. / (d + 1))
        output = _card.copy()
        greater = np.where(output > (1 / _a))
        equal = np.where(output == (1 / _a))
        lower = np.where(output < (1 / _a))
        output[greater] = (-1) ** (d + 1 - output[greater])
        output[equal] = 0
        output[lower] = (-1) ** (1 + output[lower])
        return output

    @staticmethod
    def _volume_calc(_b, _h, _mat, _svol, copula, margins):
        """
        AEP algorithm helper function.
        Volume calculator.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param _b: _b quantity AEP algorithm.
        :type _b: ``numpy.ndarray``
        :param _h: _h quantity AEP algorithm.
        :type _h: ``numpy.ndarray``
        :param _svol: _svol quantity AEP algorithm.
        :type _svol: ``numpy.ndarray``
        :param _mat: _mat quantity AEP algorithm.
        :type _mat: ``numpy.ndarray``
        :param copula: copula (dependence structure between margins).
        :type copula: ``Copula``
        :param margins: marginal distributions.
        :type margins: ``Margins``
        :return: volumes.
        :rtype: ``numpy.ndarray``
        """
        h_ = (2. / (copula.dim + 1)) * _h
        b_ = np.expand_dims(_b.T, axis=0)
        # s_ = np.array((-1) ** (copula.dim - np.sum(AEPCalculator._mat(copula.dim), axis=1))).reshape(-1, 1)
        v_ = np.hstack((b_ + h_ * _mat))
        c_ = copula.cdf(margins.cdf(v_)).reshape(-1, _b.shape[0])
        result = np.sum(c_ * (_svol * np.sign(h_) ** copula.dim), axis=0)
        return result

    @staticmethod
    def _sn_update(_sn, _msn):
        """
        AEP algorithm helper function.
        Update ``_sn`` quantity.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.
        
        :param _sn: previous ``_sn`` value.
        :type _sn: ``numpy.ndarray``
        :param _msn: _msn quantity AEP algorithm.
        :type _msn: ``numpy.ndarray``
        :param d: dimensionality of the space.
        :type d: ``int``
        :return: updated ``_sn`` value.
        :rtype: ``numpy.ndarray``
        """
        result = np.repeat(_sn, _msn.shape[0]) * np.tile(
                    _msn,
                    _sn.shape[0]
                    )
        return result

    @staticmethod
    def _h_update(_h, _card, d):
        """
        AEP algorithm helper function.
        Update ``_h`` quantity.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param _h: previous ``_h`` value.
        :type _h: ``numpy.ndarray``
        :param _card: _card quantity AEP algorithm.
        :type _card: ``numpy.ndarray``
        :param d: dimensionality of the space.
        :type d: ``int``
        :return: updated ``_h`` value.
        :rtype: ``numpy.ndarray``
        """
        result = (1 - np.tile(_card, len(_h)) * (2. / (d + 1))) * np.repeat(_h, len(_card))
        return result

    @staticmethod
    def _b_update(_b, _h, _mat, d):
        """
        AEP algorithm helper function.
        Update ``_b`` quantity.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param _b: previous ``_b`` value.
        :type _b: ``numpy.ndarray``
        :param _h: ``_h`` value.
        :type _h: ``numpy.ndarray``
        :param d: dimensionality of the space.
        :type d: ``int``
        :return: updated ``_b`` value.
        :rtype: ``numpy.ndarray``
        """
        n = _mat.shape[0]
        mat_ = _mat.transpose()
        h_ = np.repeat(_h, n).reshape(-1, 1)
        times_ = int(h_.shape[0] / n)
        result = np.repeat(_b, n, 0)
        result = result + (2. / (d + 1)) * np.tile(h_, (1, d)) * np.tile(mat_, times_).transpose()
        return result

    @staticmethod
    def core_cdf(x, n_iter, copula, margins):
        """
        AEP algorithm to approximate cdf. Non vectorized version.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``float``
        :param n_iter: number of algorithm iterations.
        :type n_iter: ``int``
        :param copula: copula (dependence structure between margins).
        :type copula: ``Copula``
        :param margins: marginal distributions.
        :type margins: ``Margins``
        :return: cumulative distribution function.
        :rtype: ``float``
        """
        # initiate quantities
        _b = np.repeat(0, copula.dim).reshape(1, copula.dim)  # Vector b of the AEP algorithm.
        _h = np.array([[x]])  # Vector h of the AEP algorithm.
        _sn = np.array([1])  # Array of +1,-1, 0 indicating whether a volume must be summed,
                             # subtracted or ignored, respectively.
        _mat = AEPCalculator._mat(copula.dim) # to be filtered later? No
        _card = np.sum(_mat, axis=1)[1:] # to be filtered later or not if created after filtered _mat
        _matvol = np.expand_dims(_mat, axis=2)
        _msn =  AEPCalculator._m(_card = _card, d=copula.dim)
        fltr = _msn != 0
        _msn = _msn[fltr, ] # filtered for efficiency
        _card = _card[fltr] # filtered for efficiency
        _svol = np.array((-1) ** (copula.dim - np.sum(_mat, axis=1))).reshape(-1, 1)
        cdf = AEPCalculator._volume_calc(_b, _h, _matvol, _svol, copula, margins)[0]
        _mat = _mat[1:, ][fltr, ] # filtered for efficiency
        _vols = 0
        # start loop. n_iter reduced by 1 as _volume_calc has already been called once.
        for _ in range(n_iter-1):
            _sn = AEPCalculator._sn_update(_sn, _msn)
            _b = AEPCalculator._b_update(_b, _h, _mat, copula.dim)
            _h = AEPCalculator._h_update(_h, _card, copula.dim)
            _vols = np.sum(_sn * AEPCalculator._volume_calc(_b, _h, _matvol, _svol, copula, margins))
            cdf += _vols
        cdf += _vols * (((copula.dim + 1) ** copula.dim) / (special.factorial(copula.dim) * 2 ** copula.dim) - 1)
        return cdf

    @staticmethod
    def cdf(x, n_iter, copula, margins):
        """
        AEP algorithm to approximate cdf.
        See Arbenz P., Embrechts P., and Puccetti G.
        "The AEP algorithm for the fast computation of the distribution of the sum of dependent random variables." Bernoulli (2011): 562-591.

        :param x: quantiles where the cumulative distribution function are evaluated.
        :type x: ``float``, ``numpy.float64`` or ``numpy.ndarray``
        :param n_iter: number of algorithm iterations.
        :type n_iter: ``int``
        :param copula: Copula (dependence structure between margins).
        :type copula: ``Copula``
        :param margins: Marginal distributions.
        :type margins: ``Margins``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        isscalar = not isinstance(x, (np.ndarray, list)) 
        x = np.ravel(x)
        output = np.empty(len(x))
        for i in range(len(output)):
            output[i] = AEPCalculator.core_cdf(x[i], n_iter, copula, margins)
        if isscalar:
            output = output.item()
        return output

    @staticmethod
    def ppf(q, n_iter, copula, margins, tol=1e-04):
        """
        Percent point function, a.k.a. the quantile function, of the random variable sum
        using AEP algorithm.
        Inverse of cumulative distribution function. See ``scipy.optimize.brentq``.
        
        :param x: quantiles where the cumulative distribution function are evaluated.
        :type x: ``float``, ``numpy.float64`` or ``numpy.ndarray``
        :param n_iter: number of AEP algorithm iterations (optional).
        :type n_iter: ``int``
        :param copula: Copula (dependence structure between margins).
        :type copula: ``Copula``
        :param margins: Marginal distributions.
        :type margins: ``Margins``
        :param tol: tolerance threshold, maximum allowed absolute difference between cumulative probability values (optional).
        :type tol: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        hf.assert_type_value(q, 'q', logger, (float, np.ndarray, list))
        isscalar = not isinstance(q, (np.ndarray, list)) 
        q = np.ravel(q)
        output = np.empty(len(q))
        for i in range(len(output)):
            output[i] = AEPCalculator.core_ppf(q[i], n_iter, copula, margins, tol)
        if isscalar:
            output = output.item()
        return output

    @staticmethod
    def core_ppf(q, n_iter, copula, margins, tol=1e-04):
        """
        Percent point function, a.k.a. the quantile function, of the random variable sum
        using AEP algorithm. Non vectorized version.
        Inverse of cumulative distribution function. See ``scipy.optimize.brentq``.
        
        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :param n_iter: number of AEP algorithm iterations (optional).
        :type n_iter: ``int``
        :param copula: Copula (dependence structure between margins).
        :type copula: ``Copula``
        :param margins: Marginal distributions.
        :type margins: ``Margins``
        :param tol: tolerance threshold, maximum allowed absolute difference between cumulative probability values (optional).
        :type tol: ``float``
        :return: percent point function.
        :rtype: ``float`` or ``numpy.float64``
        """

        q = np.ravel(q)
        qarr = np.repeat(q, copula.dim).reshape(copula.dim, -1)
        x0 = np.sum(margins.ppf(qarr))
        diff = (q - AEPCalculator.core_cdf(x0, n_iter, copula, margins))

        if abs(diff) <= tol:
            return x0
        
        if diff < 0:
            # x0 is larger than actual q-th level quantile
            x1 = 0.5 * x0
            while AEPCalculator.core_cdf(x1, n_iter, copula, margins) > q:
                # substitute x0 with x1 as it narrows the interval
                x0 = x1
                x1 = 0.5 * x1
            bracket = [x1, x0]
        else: # diff > 0:
            # x0 is smaller than actual q-th level quantile
            x1 = 2 * x0
            while AEPCalculator.core_cdf(x1, n_iter, copula, margins) < q:
                # substitute x0 with x1 as it narrows the interval
                x0 = x1
                x1 = 2 * x1
            bracket = [x0, x1]

        output = root_scalar(lambda x : q - AEPCalculator.core_cdf(x, n_iter, copula, margins), method='brentq', bracket=bracket, xtol=tol)
        if output.converged:
            return output.root
        else:
            logger.warning('Execution of ppf failed. Result does not converged')
            return np.nan

    @staticmethod
    def rvs(size, random_state, n_iter, copula, margins, tol=1e-04):
        """
        Random variates generator function of the sum of positive random variables.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``
        :param n_iter: number of AEP algorithm iterations (optional).
        :type n_iter: ``int``
        :param copula: copula (dependence structure between margins).
        :type copula: ``Copula``
        :param margins: marginal distributions.
        :type margins: ``Margins``
        :return: sample of the sum of positive random variables.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        np.random.seed(random_state) 
        u_ = np.random.uniform(size=size)
        return AEPCalculator.ppf(u_, n_iter, copula, margins, tol)
