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
    def conditions_basis_adjuster(
        layers,
        next_layer_loss,
        in_layer_loss_after_agg,
        in_layer_loss_before_agg,
        svsample,
        k
        ):
    
        if layers[k].basis == 'drop-down':
            adjusted_deductible = 0
            adjusted_cover = layers[k].cover
        elif layers[k].basis == 'regular':
            adjusted_deductible = layers[k].deductible
            adjusted_cover = layers[k].cover
            next_layer_loss = np.copy(svsample.copy)
        elif layers[k].basis == 'stretch-down':
            adjusted_deductible = 0
            filter_ = in_layer_loss_before_agg > in_layer_loss_after_agg
            adjusted_cover = layers[k].cover
            adjusted_cover[filter_] = np.sum(
                [layers[j].cover for j in range(k+1)]
                ) - in_layer_loss_after_agg[filter_]
        return (
            adjusted_cover, adjusted_deductible, next_layer_loss
        )

    @staticmethod
    def loss_maintenance_deductible_adjuster(
        layers,
        in_layer_loss_before_agg,
        in_layer_loss_after_agg,
        next_layer_loss,
        k
        ):
        condition_ = (k == 0) and (layers[k].maintenance_deductible > 0) 
        if condition_:
            filter_ = in_layer_loss_before_agg > in_layer_loss_after_agg
            filter_ = np.insert(filter_, 0, False, axis=0)
            filter_ = filter_[0:(len(filter_)-1)]
            in_layer_loss_after_agg_after_mded = np.copy(in_layer_loss_after_agg)
            in_layer_loss_after_agg_after_mded[filter_] = np.minimum(
                layers[k].maintenance_deductible,
                in_layer_loss_before_agg
            )[filter_]
            next_layer_loss = next_layer_loss - in_layer_loss_after_agg_after_mded
        return next_layer_loss, in_layer_loss_after_agg_after_mded

    @staticmethod
    def mc_simulation_tower(policystructure, frequency, severity, n_sim, random_state):
        """
        Aggregate loss distribution of tower layers via Monte Carlo simulation.

        :return: list of aggregate loss distribution nodes, empirical pdf, cdf.
        :rtype: ``list``
        """
        logger.info('..Approximating aggregate loss distribution via Monte Carlo simulation..')
        
        output = [None] * policystructure.length
        container = np.empty((policystructure.length, n_sim))
        fqsample = frequency.model.rvs(n_sim, random_state=random_state)
        svsample = severity.model.rvs(np.sum(fqsample), random_state=random_state)
        cs = np.cumsum(fqsample).astype(int)[:(n_sim-1)]
        for i in range(n_sim-1):
            next_layer_loss = np.split(svsample, cs[i])
            in_layer_loss_after_agg = np.zeros(next_layer_loss.shape[0])
            in_layer_loss_before_agg = np.zeros(next_layer_loss.shape[0])
            
            for k in range(policystructure.length):
                adjusted_conds = LossModelCalculator.conditions_basis_adjuster(
                    policystructure.layers,
                    next_layer_loss,
                    in_layer_loss_after_agg,
                    in_layer_loss_before_agg,
                    svsample,
                    k
                )
                adjusted_cover, adjusted_deductible, next_layer_loss = adjusted_conds
            
                in_layer_loss_before_agg = np.minimum(
                    np.maximum(next_layer_loss - adjusted_deductible, 0),
                    adjusted_cover
                )
                in_layer_loss_after_agg = np.diff(
                    np.minimum(policystructure.layers[k].aggr_cover,
                    np.cumsum(in_layer_loss_before_agg))
                )
                # adjust next layer loss and in_layer_loss_after_agg
                # in case of maintenance deductible
                adjusted_losses = LossModelCalculator.loss_maintenance_deductible_adjuster(
                    in_layer_loss_before_agg,
                    in_layer_loss_after_agg,
                    next_layer_loss,
                    k
                    )

                # next layer goes to the next k iteration
                next_layer_loss, in_layer_loss_after_agg = adjusted_losses
                container[k, i] = np.sum(in_layer_loss_after_agg)
            # finally adjust retention loss
            container[0, i] = np.sum(svsample) - np.sum(container[1:policystructure.length, i])

        logger.info('..Simulation completed..')

        for k in range(policystructure.length):
            x = container[k, :]
            x_ = np.unique(container[k, :])
            cdf_ = hf.ecdf(x)(x_)
            output[k] = {
                'cdf': cdf_,
                'nodes': x_
                }

        return output

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
