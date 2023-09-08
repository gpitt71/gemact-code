from .libraries import *
from . import config
from . import helperfunctions as hf
from . import distributions as distributions
from .calculators import LossModelCalculator as Calculator

quick_setup()
logger = log.name('lossmodel')

class PolicyStructure:
    """
    Policy structure component of a loss model.

    :param layers: Non-proportional layer (default is infinity-xs-0 layer).
    :type layers: ``Layer``, ``LayerTower``, ``list``
    """

    def __init__(self, layers=None):
        self.layers = layers if layers is not None else Layer()

    @property
    def layers(self):
        return self.__layers

    @layers.setter
    def layers(self, value):
        hf.assert_type_value(
            value, 'layers',
            type=(Layer, list), logger=logger
            )
        if isinstance(value, Layer):
            value = [value]
            # logger.info('layers converted to list for internal consistency')
        elif isinstance(value, list):
            all(hf.assert_type_value(
                item, 'layers item', type=Layer, logger=logger
                ) for item in value)
            # value = LayerTower(*value)
        self.__layers = value
    
    @property
    def length(self):
        return len(self.layers)

    def index_to_layer_name(self, idx):
        """
        Return name of a layer given its index.
        """
        message = 'ValueError.\n No Layer has index %s.' % (idx)
        try:
            output = self.layers[idx].name
        except ValueError as e:
            logger.error(message)
            e.args += (message, )
            raise
        return output
    
    def layer_name_to_index(self, name):
        """
        Return index of a layer given its name.
        """
        output = [idx for idx in range(self.length) if self.layers[idx].name == name][0]
        if not output:
            message = 'ValueError. No Layer has name "%s".' % (name)
            logger.error(message)
            raise ValueError(message)
        return output

class Layer:
    """
    Policy structure non-proportional layer.

    :param deductible: each-and-every-loss (non-ranking) deductible, also referred to as retention or priority (default value is 0).
    :type deductible: ``int`` or ``float``
    :param cover: each-and-every-loss cover, also referred to as limit (default value is infinity). Cover plus deductible is the upper priority or severity 'exit point'.
    :type cover: ``int`` or ``float``
    :param aggr_deductible: aggregate deductible (default value is 0). 
    :type aggr_deductible: ``int`` or ``float``
    :param aggr_cover: aggregate cover, also referred to as aggregate limit (default is infinity).
    :type aggr_cover: ``int`` or ``float``
    :param n_reinst: Number of reinstatements (default value is infinity).
                    When reinstatements are free (percentage = 0), an alternative parametrization is aggregate cover = (number of reinstatement + 1) * cover.
                    E.g. When the number of reinstatements = 0, the aggregate cover is equal to the cover,
                    when number of reinstatements is infinity there is no aggregate cover
                    (the aggregate cover is infinity).
    :type n_reinst: ``int``
    :param reinst_percentage: percentage of reinstatements layers (default value is 0), typically a value in [0, 1].
    :type reinst_percentage: ``int`` or ``float`` or ``np.array``
    :param maintenance_deductible: maintenance deductible, sometimes referred to as residual each-and-every-loss deductible (default is 0). Non-zero maintenance deductible applies to retention layers only.
    :type maintenance_deductible: ``int`` or ``float``
    :param share: Partecipation share of the layer (default is 1).
    :type share: ``float``
    :param basis: layer basis (default is 'regular'). One of 'regular', 'drop-down', 'stretch-down'. 
    :type basis: ``str``
    """

    def __init__(
        self,
        cover=float('inf'),
        deductible=0,
        aggr_cover=float('inf'),
        aggr_deductible=0,
        n_reinst=float('inf'),
        reinst_percentage=0,
        maintenance_deductible=0,
        share=1,
        basis='regular',
        ):

        self.cover = cover
        self.deductible = deductible
        self.aggr_deductible = aggr_deductible
        self.aggr_cover = aggr_cover
        self.n_reinst = n_reinst
        self.reinst_percentage = reinst_percentage
        self.maintenance_deductible = maintenance_deductible
        self.share = share
        self.basis = basis
        self._check_and_set_category()

    @property
    def name(self):
        output = ('%s-xs-%s, %s-xs-%s in aggr, %s share' %(
            self.cover,
            self.deductible,
            self.aggr_cover,
            self.aggr_deductible,
            self.share
            ))
        return output

    @property
    def deductible(self):
        return self.__deductible

    @deductible.setter
    def deductible(self, value):
        name = 'deductible'
        hf.assert_type_value(value, name, logger=logger, type=(int, float),
        lower_bound=0, upper_bound=float('inf'), upper_close=False)
        self.__deductible = float(value)

    @property
    def cover(self):
        return self.__cover

    @cover.setter
    def cover(self, value):
        name = 'cover'
        hf.assert_type_value(value, name, logger, type=(int, float),
        lower_bound=0, lower_close=False)
        self.__cover = float(value)

    @property
    def aggr_deductible(self):
        return self.__aggr_deductible

    @aggr_deductible.setter
    def aggr_deductible(self, value):
        name = 'aggr_deductible'
        hf.assert_type_value(value, name, logger, type=(int, float),
        lower_bound=0, upper_bound=float('inf'), upper_close=False)
        self.__aggr_deductible = value

    @property
    def n_reinst(self):
        return self.__n_reinst

    @n_reinst.setter
    def n_reinst(self, value):
        name = 'n_reinst'
        if value is not None:
            hf.assert_type_value(
                value, name, logger,
                type=(int, float, np.inf, np.floating), lower_bound=0
                )
        self.__n_reinst = value

    @property
    def reinst_percentage(self):
        return self.__reinst_percentage

    @reinst_percentage.setter
    def reinst_percentage(self, value):
        name = 'reinst_percentage'
        name_two = 'reinst_percentage size'

        if value is not None:
            if self.n_reinst == 0 or not self.n_reinst < float('inf'):
                # logger.info('reinst_percentage set to None.')
                value = None
            else:
                hf.assert_type_value(
                    value, name, logger, type=(int, float, np.ndarray, list, tuple)
                    )
                if isinstance(value, (int, float)):
                    hf.assert_type_value(
                    value, name, logger, type=(int, float, np.floating),
                    lower_bound=0, upper_bound=1
                    )
                    value = np.repeat(value, self.n_reinst)
                else:
                    value = np.asarray(value).ravel()
                    for val in value:
                        hf.assert_type_value(
                        val, name, logger, type=(int, float, np.floating),
                        lower_bound=0, upper_bound=1
                        )
                    hf.check_condition(
                        value=value.shape[0],
                        check=self.n_reinst,
                        name=name_two,
                        logger=logger,
                        type='=='
                        )
        self.__reinst_percentage = value

    @property
    def aggr_cover(self):
        return self.__aggr_cover
            
    @aggr_cover.setter
    def aggr_cover(self, value):
        name = 'aggr_cover'
        # if value is not None:
        hf.assert_type_value(
            value, name, logger,
            type=(int, float, np.inf, np.floating), lower_bound=0
            )
        self.__aggr_cover = value

    @property
    def manteinance_deductible(self):
        return self.__manteinance_deductible   

    @manteinance_deductible.setter
    def manteinance_deductible(self, value):
        name = 'maintenance_deductible'
        hf.assert_type_value(value, name, logger=logger, type=(int, float),
        lower_bound=0, upper_bound=float('inf'), upper_close=False)
        if self.deductible > 0 and value > 0:
            value = 0
            logger.warning('Manteinance deductible applies to retention layer only (deductible = 0), manteinance_deductible set to 0.')
        self.__manteinance_deductible = value

    @property
    def share(self):
        return self.__share

    @share.setter
    def share(self, value):
        hf.assert_type_value(
            value, 'share', type=(float, int), logger=logger,
            upper_bound=1, upper_close=True,
            lower_bound=0, lower_close=True
            )
        self.__share = value

    @property
    def basis(self):
        return self.__basis

    @basis.setter
    def basis(self, value):
        hf.assert_member(value, config.POLICY_LAYER_BASIS, logger)
        if value not in ('regular'):
            logger.warning('Currently, basis is treated as "regular".')
        self.__basis = value

    @property
    def category(self):
        return self.__category

    @property
    def exit_point(self):
        return self.deductible + self.cover
    
    @property
    def identifier(self):
        # share not included
        # output = '{}_{}_{}_{}_{}_{}_{}'.format(
        output = '{}_{}_{}_{}_{}_{}'.format(
            self.deductible,
            self.cover,
            self.aggr_deductible,
            self.aggr_cover,
            self.n_reinst,
            self.reinst_percentage
            # self.share
            )
        return output
    
    @staticmethod
    def specs():
        """
        Method (static) returning layer specifications names.

        :return: layer specifications names.
        :rtype: ``set``
        """
        return {
            'deductible', 'cover', 'aggr_deductible',
            'aggr_cover', 'n_reinst', 'reinst_percentage',
            'maintenance_deductible', 'share',
            'basis'
            }
    
    def _check_presence_aggr_conditions(self):
        """
        Method that check if aggregate conditions are present.

        :return: True if aggregate conditions are present, else False.
        :rtype: ``bool``
        """
        # xlrs case
        output = False
        if self.__category == 'xlrs':
            output = True
        else:
            if self.aggr_deductible > 0 or self.aggr_cover < np.infty:
                output = True
        return output

    def _check_and_set_category(self):
        """
        Method that check and set the category of the layer.

        :return: Void.
        :rtype: ``None``
        """
        # xlrs case
        if self.n_reinst is not None and np.isfinite(self.n_reinst):
            hf.check_condition(
                    value=self.cover,
                    check=float('inf'),
                    name='cover',
                    logger=logger,
                    type='!='
                )
            self.aggr_cover = self.cover * (self.n_reinst + 1) 
            self.__category = 'xlrs'
        else: # non 'xlrs' cases
            self.n_reinst = None
            self.reinst_percentage = None
            self.__category = 'xl/sl'
        # final check
        hf.assert_member(
            self.__category, config.POLICY_LAYER_CATEGORY, logger
        )
        return

class Frequency:
    """
    Frequency component of the loss models underlying the collective risk model.

    :param dist: name of the frequency distribution.
    :type dist: ``str``
    :param par: parameters of the frequency distribution.
    :type par: ``dict``
    :param threshold: analysis threshold where the frequency model refers (optional). Default is 0, i.e. the 'ground up' or reporting threshold frequency.
                See pag. 323 in Parodi, P. (2014). Pricing in general insurance (first ed.). 
    :type threshold: ``int`` or ``float``
    """

    def __init__(self, dist, par, threshold=0):
        self.dist = dist
        self.par = par
        self.threshold = threshold
        self.model = self.dist(**self.par)

    @property
    def dist(self):
        return self.__dist

    @dist.setter
    def dist(self, value):
        hf.assert_member(value, config.DIST_DICT, logger, config.SITE_LINK)
        self.__dist = eval(config.DIST_DICT[value])

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, value):
        hf.assert_type_value(value, 'par', logger, dict)
        
        try:
            self.dist(**value)
        except Exception:
            logger.error('Wrong parameters in par \n '
                         'Make sure that dist is correctly parametrized.\n See %s' % config.SITE_LINK)
            raise

        if 'zm' in self.dist(**value).category() and 'loc' not in value.keys():
            value['loc'] = -1

        self.__par = value

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, value):
        hf.assert_type_value(
            value, 'threshold', logger, (int, float),
            lower_bound=0, lower_close=True
            )
        self.__threshold = value
        return

    @property
    def p0(self):
        output = self.par['p0M'] if 'zm' in self.model.category() else None
        return output

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        hf.assert_member('frequency', value.category(), logger, config.SITE_LINK)
        self.__model = value

    def abp0g0(self, fj):
        """
        Parameters of the frequency distribution according to the (a, b, k) parametrization,
        the probability generating function computed in zero given the discrete severity probs,
        and the probability of the distribution in zero.

        :param fj: discretized severity distribution probabilities.
        :type fj: ``numpy.ndarray``

        :return: a, b, probability in zero and aggregate cost probability in zero.
        :rtype: ``tuple``
        """
        a, b, p0 = self.model.abk()
        return a, b, p0, [self.model.pgf(fj[0])]

class Severity:
    """
    Severity component of the loss models underlying the collective risk model.
    Severity model is always considered to start at (the relative) 0, i.e. the reporting threshold.

    :param dist: name of the frequency distribution.
    :type dist: ``str``
    :param par: parameters of the frequency distribution.
    :type par: ``dict``
    """

    def __init__(self, dist, par):
        self.dist = dist
        self.par = par
        self.model = self.dist(**self.par)
        self.loc = self.par['loc'] if 'loc' in self.par.keys() else .0
        
    @property
    def dist(self):
        return self.__dist

    @dist.setter
    def dist(self, value):
        hf.assert_member(value, config.DIST_DICT, logger, config.SITE_LINK)
        self.__dist = eval(config.DIST_DICT[value])

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, value):
        hf.assert_type_value(value, 'par', logger, dict)
        try:
            self.dist(**value)
        except Exception:
            logger.error('Wrong parameters in par \n '
                         'Please make sure that dist is correctly parametrized.\n See %s' % config.SITE_LINK)
            raise
        self.__par = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (int, float))
        self.__loc = value

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        hf.assert_member('severity', value.category(), logger, config.SITE_LINK)
        self.__model = value
    
    def excess_frequency(self, x, base_frequency=100):
        """
        Expected excess frequency function, i.e. expected frequency in excess of a given threshold.

        :param x: value where excess frequency is evaluated.
        :type x: ``float``
        :param base_frequency: frequency at origin (default is 100). Optional.
        :type base_frequency: ``int``, ``float``
        :return: excess frequency.
        :rtype: ``numpy.float`` or ``float``
        """
        hf.assert_type_value(x, 'x', logger, (float, int, np.floating), lower_bound=0, lower_close=False)
        return base_frequency * self.model.sf(x)

    def return_period(self, x, base_frequency=100):
        """
        Expected return period, given a base frequency.

        :param x: value whose return period is evaluated.
        :type x: ``float``
        :param base_frequency: frequency at origin (default is 100). Optional.
        :type base_frequency: ``int``, ``float``
        :return: return period.
        :rtype: ``numpy.float`` or ``float``
        """
        hf.assert_type_value(x, 'x', logger, (float, int, np.floating), lower_bound=0, lower_close=False)
        return 1 / self.excess_frequency(x, base_frequency) 

    def censored_var(self, cover, deductible):
        """
        Variance of the transformed severity min(max(x - u, 0), v).
                
        :param cover: cover, also referred to as limit. cover plus deductible is the upper priority or severity 'exit point'.
        :type cover: ``int``, ``float``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int``, ``float``
        :return: variance of the transformed severity.
        :rtype: ``numpy.float``
        """
        output = self.model.censored_moment(n=2, u=deductible, v=cover) - \
                    self.censored_mean(deductible=deductible, cover=cover)**2
        return output

    def censored_std(self, cover, deductible):
        """
        Standard deviation of the transformed severity min(max(x - u, 0), v).
                
        :param cover: cover, also referred to as limit. cover plus deductible is the upper priority or severity 'exit point'.
        :type cover: ``int``, ``float``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int``, ``float``
        :return: standard deviation of the transformed severity.
        :rtype: ``numpy.float``
        """
        return self.censored_var(deductible=deductible, cover=cover)**(1/2)

    def censored_mean(self, cover, deductible):
        """
        Mean of the transformed severity min(max(x - u, 0), v).
        Also referred to as the stop-loss transformation function.
                
        :param cover: cover, also referred to as limit. cover plus deductible is the upper priority or severity 'exit point'.
        :type cover: ``int``, ``float``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int``, ``float``
        :return: mean of the transformed severity.
        :rtype: ``numpy.float``
        """
        return self.model.censored_moment(n=1, u=deductible, v=cover)
    
    def censored_skewness(self, cover, deductible):
        """
        Skewness of the transformed severity min(max(x - u, 0), v).
                
        :param cover: cover, also referred to as limit. cover plus deductible is the upper priority or severity 'exit point'.
        :type cover: ``int``, ``float``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int``, ``float``
        :return: skewness of the transformed severity.
        :rtype: ``numpy.float``
        """
        num1 = self.model.censored_moment(n=3, u=deductible, v=cover)
        num2 = 3 * self.censored_mean(deductible=deductible, cover=cover) * self.censored_var(deductible=deductible, cover=cover)
        num3 = self.censored_mean(deductible=deductible, cover=cover) ** 3
        den = self.censored_std(deductible=deductible, cover=cover) ** 3
        return (num1 - num2 - num3)/den

    def censored_coeff_variation(self, cover, deductible):
        """
        Coefficient of variation (CoV) of the transformed severity min(max(x - u, 0), v).
                
        :param cover: cover, also referred to as limit. cover plus deductible is the upper priority or severity 'exit point'.
        :type cover: ``int``, ``float``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int``, ``float``
        :return: CoV of the transformed severity.
        :rtype: ``numpy.float``
        """
        return self.censored_std(deductible=deductible, cover=cover) / self.censored_mean(deductible=deductible, cover=cover)

    def discretize(
        self,
        discr_method,
        n_discr_nodes,
        discr_step,
        deductible=0
        ):
        """
        Severity discretization according to the discretization method selected by the user.

        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int`` or ``float``
        :param discr_method: severity discretization method. One of 'massdispersal', 'localmoments',
                             'upperdiscretization', 'lowerdiscretization'.
        :type discr_method: ``str``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :param n_discr_nodes: number of nodes of the discretized severity. Optional, default is 0.
        :type n_discr_nodes: ``int``
        :return: discrete severity, nodes sequence and discrete probabilities.
        :rtype: ``dict``
        """
        
        hf.assert_member(discr_method, config.SEV_DISCRETIZATION_METHOD, logger)
        hf.assert_type_value(deductible, 'deductible', logger, type=(int, float), lower_bound=0)
        # hf.assert_type_value(cover, 'cover', logger, type=(int, float), lower_bound=0, lower_close=False)
        hf.assert_type_value(n_discr_nodes, 'n_discr_nodes', logger, type=(int, float), lower_bound=1)
        n_discr_nodes = int(n_discr_nodes)
        hf.assert_type_value(discr_step, 'discr_step', logger, type=(int, float), lower_bound=0, lower_close=False)
        discr_step = float(discr_step)
        
        if discr_method == 'massdispersal':
            return Calculator.mass_dispersal(
                self,
                deductible,
                discr_step,
                n_discr_nodes
                )
        elif discr_method == 'localmoments':
            return Calculator.local_moments(
                self,
                deductible,
                discr_step,
                n_discr_nodes
                )

        elif discr_method == 'lowerdiscretization':
            return Calculator.lower_discretization(
                self,
                deductible,
                discr_step,
                n_discr_nodes
                )

        elif discr_method == 'upperdiscretization':
            return Calculator.upper_discretization(
                self,
                deductible,
                discr_step,
                n_discr_nodes
                )
    
    def plot_discr_sev_cdf(
            self,
            discr_method,
            n_discr_nodes,
            discr_step,
            deductible,
            log_x_scale=False,
            log_y_scale=False,
            **kwargs
            ):
        """
        Plot the cumulative distribution function of the discretized severity distribution.

        :param discr_method: severity discretization method. One of 'massdispersal', 'localmoments',
                                'upperdiscretization', 'lowerdiscretization'.
        :type discr_method: ``str``
        :param n_discr_nodes: number of nodes of the discretized severity.
        :type n_discr_nodes: ``int``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int`` or ``float``
        :param log_x_scale: if ``True`` the x-axis scale is logarithmic (optional).
        :type log_x_scale: ``bool``
        :param log_y_scale: if ``True`` the y-axis scale is logarithmic (optional).
        :type log_y_scale: ``bool``

        :param \\**kwargs:
            Additional parameters as those for ``matplotlib.axes.Axes.step``.

        :return: plot of the cdf.
        :rtype: ``matplotlib.figure.Figure``
        """

        hf.assert_type_value(log_x_scale, 'log_x_scale', logger, bool)
        hf.assert_type_value(log_y_scale, 'log_y_scale', logger, bool)

        sevdict = self.discretize(
            discr_method=discr_method,
            n_discr_nodes=n_discr_nodes,
            discr_step=discr_step,
            deductible=deductible
        )

        x_ = sevdict['nodes']
        y_ = np.cumsum(sevdict['fj'])

        figure = plt.figure()
        ax = figure.add_subplot(111)

        ax.step(x_, y_, '-', where='post', **kwargs)
        if log_y_scale:
            ax.set_yscale('log')
        if log_x_scale:
            ax.set_xscale('log')
        ax.set_title('Discretized severity cumulative distribution function')
        ax.set_ylabel('cdf')
        ax.set_xlabel('nodes')
        return ax

class LossModel:
    """
    Loss model for (re)insurance costing and risk modeling using a collective risk model framework.

    :param severity: severity model.
    :type severity: ``Severity``
    :param frequency: frequency model.
    :type frequency: ``Frequency``
    :param policystructure: policy structure.
    :type policystructure: ``PolicyStructure``
    :param aggr_loss_dist_method: computational method to approximate the aggregate loss distribution.
                                  One of Fast Fourier Transform ('fft'),
                                  Panjer recursion ('recursion') and Monte Carlo simulation ('mc').
    :type aggr_loss_dist_method: ``str``
    :param n_sim: number of simulations of Monte Carlo (mc) method for the aggregate loss distribution approximation.
    :type n_sim: ``int``
    :param tilt: whether tilting of fft is present or not (default is 0).
    :type tilt: ``bool``
    :param tilt_value: tilting parameter value of fft method for the aggregate loss distribution approximation.
    :type tilt_value: ``float``
    :param random_state: random state for the random number generator in mc.
    :type random_state: ``int``
    :param n_aggr_dist_nodes: number of nodes in the approximated aggregate loss distribution. It cannot be lower than 256.
    :type n_aggr_dist_nodes: ``int``
    :param sev_discr_method: severity discretization method. One of 'massdispersal', 'localmoments',
                            'upperdiscretization', 'lowerdiscretization'.
    :type sev_discr_method: ``str``
    :param n_sev_discr_nodes: number of nodes of the discretized severity (optional).
    :type n_sev_discr_nodes: ``int``
    :param sev_discr_step: severity discretization step.
    :type sev_discr_step: ``float``
    """

    def __init__(
        self,
        severity,
        frequency,
        policystructure=PolicyStructure(),
        aggr_loss_dist_method=None,
        n_sim=10000,
        tilt=False,
        tilt_value=0,
        random_state=None,
        n_aggr_dist_nodes=20000,
        sev_discr_method='localmoments',
        n_sev_discr_nodes=None,
        sev_discr_step=None,
        ):

        # Frequency, Severity and PolicyStructure
        self.frequency = frequency
        self.severity = severity
        self.policystructure = policystructure
        # Severity discretization parameters
        self.sev_discr_method = sev_discr_method
        self.n_sev_discr_nodes = n_sev_discr_nodes
        self.sev_discr_step = sev_discr_step
        # Aggregate loss distribution specifications
        self.aggr_loss_dist_method = aggr_loss_dist_method
        self.n_sim = n_sim
        self.random_state = random_state
        self.n_aggr_dist_nodes = n_aggr_dist_nodes
        self.tilt = tilt
        self.tilt_value = tilt_value
        # Initializing calculation for aggregate loss distribution 
        self.__dist = [None] * self.policystructure.length
        self.__dist_excl_aggr_cond = [None] * self.policystructure.length
        self.__pure_premium = [None] * self.policystructure.length
        self.__pure_premium_dist = [None] * self.policystructure.length
        self.dist_calculate()
        self.costing()

    @property
    def severity(self):
        return self.__severity

    @severity.setter
    def severity(self, value):
        hf.assert_type_value(value, 'severity', logger, Severity)
        self.__severity = value

    @property
    def frequency(self):
        return self.__frequency

    @frequency.setter
    def frequency(self, value):
        hf.assert_type_value(value, 'frequency', logger, Frequency)
        self.__frequency = value

    @property
    def policystructure(self):
        return self.__policystructure

    @policystructure.setter
    def policystructure(self, value):
        hf.assert_type_value(value, 'policystructure', logger, PolicyStructure)
        self.__policystructure = value

    @property
    def aggr_loss_dist_method(self):
        return self.__aggr_loss_dist_method

    @aggr_loss_dist_method.setter
    def aggr_loss_dist_method(self, value):
        if value is not None:
            hf.assert_member(value, config.AGGREGATE_LOSS_APPROX_METHOD, logger)
        self.__aggr_loss_dist_method = value

    @property
    def n_sim(self):
        return self.__n_sim

    @n_sim.setter
    def n_sim(self, value):
        hf.assert_type_value(value, 'n_sim', logger, (float, int), lower_bound=1)
        value = int(value)
        self.__n_sim = value

    @property
    def random_state(self):
        return self.__random_state

    @random_state.setter
    def random_state(self, value):
        self.__random_state = hf.handle_random_state(value, logger)

    @property
    def n_aggr_dist_nodes(self):
        return self.__n_aggr_dist_nodes

    @n_aggr_dist_nodes.setter
    def n_aggr_dist_nodes(self, value):
        hf.assert_type_value(value, 'n_aggr_dist_nodes', logger, (float, int), lower_bound=1)
        value = int(value)
        self.__n_aggr_dist_nodes = value

    @property
    def tilt(self):
        return self.__tilt

    @tilt.setter
    def tilt(self, value):
        hf.assert_type_value(value, 'tilt', logger, bool)
        self.__tilt = value

    @property
    def tilt_value(self):
        return self.__tilt_value

    @tilt_value.setter
    def tilt_value(self, value):
        hf.assert_type_value(value, 'tilt_value', logger, (float, int), lower_bound=0)
        self.__tilt_value = value

    @property
    def sev_discr_method(self):
        return self.__sev_discr_method

    @sev_discr_method.setter
    def sev_discr_method(self, value):
        self.__sev_discr_method = value

    @property
    def n_sev_discr_nodes(self):
        return self.__n_sev_discr_nodes

    @n_sev_discr_nodes.setter
    def n_sev_discr_nodes(self, value):
        self.__n_sev_discr_nodes = value

    @property
    def sev_discr_step(self):
        return self.__sev_discr_step

    @sev_discr_step.setter
    def sev_discr_step(self, value):
        self.__sev_discr_step = value

    @property
    def dist(self):
        return self.__dist
    
    @property
    def pure_premium(self):
        return self.__pure_premium
    
    @property
    def pure_premium_dist(self):
        return self.__pure_premium_dist

    def dist_calculate(
        self,
        aggr_loss_dist_method=None,
        n_aggr_dist_nodes=None,
        n_sim=None,
        random_state=None,
        sev_discr_method=None,
        sev_discr_step=None,
        n_sev_discr_nodes=None,
        tilt=None,
        tilt_value=None
        ):
        """
        Approximate the aggregate loss distributions of each policystructure layer.
        Distributions can be accessed via the ``dist`` property,
        which is a list of ``distributions.PWC`` objects, each one representing a aggregate loss distribution.

        :param aggr_loss_dist_method: computational method to approximate the aggregate loss distribution.
                                      One of Fast Fourier Transform ('fft'), Panjer recursion ('recursion')
                                      and Monte Carlo simulation ('mc'), optional (default 'mc').
        :type aggr_loss_dist_method: ``str``
        :param n_aggr_dist_nodes: number of nodes in the approximated aggregate loss distribution.
                                  Remark: before application of eventual aggregate conditions.
        :type n_aggr_dist_nodes: ``int``
        :param n_sim: number of simulations of Monte Carlo (mc) method
                      for the aggregate loss distribution approximation, optional (default is 10000).
        :type n_sim: ``int``
        :param random_state: random state for the random number generator in MC, optional.
        :type random_state: ``int``
        :param sev_discr_method: severity discretization method, optional (default is 'localmoments').
        :type sev_discr_method: ``str``
        :param sev_discr_step: severity discretization step.
        :type sev_discr_step: ``float``
        :param n_sev_discr_nodes: number of nodes of the discretized severity.
        :type n_sev_discr_nodes: ``int``
        :param tilt: whether tilting of fft is present or not, optional (default is 0).
        :type tilt: ``bool``
        :param tilt_value: tilting parameter value of fft method for the aggregate loss distribution approximation,
                           optional.
        :type tilt_value: ``float``
        :return: void
        :rtype: ``None``
        """

        if (aggr_loss_dist_method is None) and (self.aggr_loss_dist_method is None):
            logger.warning('Aggregate loss distribution calculation is omitted as aggr_loss_dist_method is missing')
            return
        
        for lay in self.policystructure.layers:
            lay._check_and_set_category()    

        verbose = True if self.policystructure.length > 1 else False
        aggr_dist_list_incl_aggr_cond = [None] * self.policystructure.length
        aggr_dist_list_excl_aggr_cond = [None] * self.policystructure.length
        if verbose:
            logger.info('Computation of layers started')
        verbose = True if self.policystructure.length > 1 else False
        for i in range(self.policystructure.length):
            if verbose:
                logger.info('Computing layer: %s' %(i+1))
            layer = self.policystructure.layers[i]
            
            # adjust frequency model from threshold to the deductible.
            factor = self.severity.model.sf(layer.deductible)/\
                self.severity.model.sf(self.frequency.threshold)
            if factor < 1:
                self.frequency.model.par_deductible_adjuster(factor)
            elif factor > 1:
                message = 'Deductible of layer %s is less than your frequency threshold.' %(i+1)
                logger.warning(message)
                self.frequency.model.par_deductible_reverter(1/factor)

            if aggr_loss_dist_method is not None:
                self.aggr_loss_dist_method = aggr_loss_dist_method
            hf.assert_not_none(
                value=self.aggr_loss_dist_method,
                name='aggr_loss_dist_method',
                logger=logger
            )

            if self.aggr_loss_dist_method == 'mc':
                if n_sim is not None:
                    self.n_sim = n_sim
                hf.assert_not_none(
                    value=self.n_sim,
                    name='n_sim',
                    logger=logger
                )
                if random_state is not None:
                    self.random_state = random_state
                # Remark: no assert_not_none needed since
                # self.random_state cannot be None due to hf.handle_random_state
                
                logger.info('Approximating aggregate loss distribution via Monte Carlo simulation')
                aggr_dist_excl_aggr_cond = Calculator.mc_simulation(
                    severity=self.severity,
                    frequency=self.frequency,
                    cover=layer.cover,
                    deductible=layer.deductible,
                    n_sim=self.n_sim,
                    random_state=self.random_state
                    )
                logger.info('MC simulation completed')
                
            else:
                if sev_discr_method is not None:
                    self.sev_discr_method = sev_discr_method
                if sev_discr_step is not None:
                    self.sev_discr_step = sev_discr_step
                if n_sev_discr_nodes is not None:
                    self.n_sev_discr_nodes = n_sev_discr_nodes
                if n_aggr_dist_nodes is not None:
                    self.n_aggr_dist_nodes = n_aggr_dist_nodes
                hf.assert_not_none(
                    value=self.sev_discr_method,
                    name='sev_discr_method',
                    logger=logger
                )
                hf.assert_type_value(
                    value=self.n_aggr_dist_nodes,
                    name='n_aggr_dist_nodes',
                    logger=logger,
                    type=(float, int, np.floating, np.integer),
                    lower_bound=2**8,
                    lower_close=True
                )

                if self.n_sev_discr_nodes is None:
                        self.n_sev_discr_nodes = self.n_aggr_dist_nodes // (2**3)

                if layer.cover == float('inf'):
                    hf.assert_not_none(
                        value=self.sev_discr_step,
                        name='sev_discr_step',
                        logger=logger
                    )
                else:
                    self.sev_discr_step = (layer.cover) / (self.n_sev_discr_nodes-1)
                    
                hf.check_condition(
                    value=self.n_aggr_dist_nodes,
                    check=self.n_sev_discr_nodes,
                    name='n_aggr_dist_nodes',
                    logger=logger,
                    type='>='
                )

                sevdict = self.severity.discretize(
                    discr_method=self.sev_discr_method,
                    n_discr_nodes=self.n_sev_discr_nodes,
                    discr_step=self.sev_discr_step,
                    deductible=layer.deductible
                    )
                
                if self.aggr_loss_dist_method == 'recursion':

                    logger.info('Approximating aggregate loss distribution via Panjer recursion')
                    aggr_dist_excl_aggr_cond = Calculator.panjer_recursion(
                        severity=sevdict,
                        discr_step=self.sev_discr_step,
                        n_aggr_dist_nodes=self.n_aggr_dist_nodes,
                        frequency=self.frequency
                        )
                    logger.info('Panjer recursion completed')

                else:  # self.aggr_loss_dist_method == 'fft'
                    if tilt is not None:
                        self.tilt = tilt
                    if tilt_value is not None:
                        self.tilt_value = tilt_value

                    hf.assert_not_none(
                        value=self.tilt_value,
                        name='tilt_value',
                        logger=logger
                    )
                    hf.assert_not_none(
                        value=self.tilt,
                        name='tilt',
                        logger=logger
                    )

                    logger.info('Approximating aggregate loss distribution via FFT')
                    aggr_dist_excl_aggr_cond = Calculator.fast_fourier_transform(
                        severity=sevdict,
                        discr_step=self.sev_discr_step,
                        tilt=self.tilt,
                        tilt_value=self.tilt_value,
                        frequency=self.frequency,
                        n_aggr_dist_nodes=self.n_aggr_dist_nodes
                        )
                    logger.info('FFT completed') 

            # restore original unadjusted frequency model
            if factor < 1:
                self.frequency.model.par_deductible_reverter(factor)
            elif factor > 1:
                self.frequency.model.par_deductible_adjuster(1/factor)

            aggr_dist_list_excl_aggr_cond[i] = distributions.PWC(
                    nodes=layer.share*aggr_dist_excl_aggr_cond['nodes'],
                    cumprobs=aggr_dist_excl_aggr_cond['cdf'],
                    legit=False
                )

            aggr_dist_incl_aggr_cond = self._apply_aggr_conditions(
                dist=aggr_dist_excl_aggr_cond,
                deductible=layer.aggr_deductible,
                cover=layer.aggr_cover
                )
            aggr_dist_list_incl_aggr_cond[i] = distributions.PWC(
                    nodes=layer.share*aggr_dist_incl_aggr_cond['nodes'], #inodes,
                    cumprobs=aggr_dist_incl_aggr_cond['cdf'], # icumprobs
                    legit=False
                )
            
            # go next i
        if verbose:
            logger.info('Computation of layers completed')
        self.__dist_excl_aggr_cond = aggr_dist_list_excl_aggr_cond
        self.__dist = aggr_dist_list_incl_aggr_cond
        return

    def _apply_aggr_conditions(self, dist, cover, deductible):
        """
        Apply aggregate conditions, i.e. aggregate deductble and aggregate cover, to a aggregate loss distribution.

        :param dist: aggregate loss distribution (before aggregate conditions).
        :type dist: ``dict``
        :param cover: (aggregate) cover.
        :type cover: ``int`` or ``float``
        :param deductible: (aggregate) deductible.
        :type deductible: ``int`` or ``float``
        :return: aggregate loss distribution after aggregate conditions.
        :rtype: ``dict``
        """
        output = dict()
        deductible = np.array(deductible).ravel()
        cover = np.array(cover).ravel()
        nodes = hf.layerFunc(
            nodes=dist['nodes'],
            cover=cover,
            deductible=deductible
            ).ravel()
        
        start_idx = (np.argmin(nodes == nodes[0]) - 1)
        end_idx = (np.argmax(nodes == nodes[-1]))
        output['nodes'] = nodes[start_idx:(end_idx + 1)]
        output['cdf'] = np.append(
            dist['cdf'][start_idx:end_idx],
            dist['cdf'][-1]
        )      
        
        return output

    def moment(self, central=False, n=1, idx=0):
        """
        Approximated aggregate loss distribution moment of order n.
        It is based on ``dist`` property.

        :param central: ``True`` if the moment is central, ``False`` if the moment is raw.
        :type central: ``bool``
        :param n: order of the moment, optional (default is 1).
        :type n: ``int``
        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: moment of order n.
        :rtype: ``numpy.float64``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        if self._check_missing_dist(idx):
            return None
        else:
            return self.dist[idx].moment(central, n)

    def ppf(self, q, idx=0):
        """
        Aggregate loss distribution percent point function, a.k.a. the quantile function,
        inverse of the cumulative distribution function.

        :param q: probability.
        :type q: ``float`` or ``numpy.ndarray``
        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: quantile.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        if self._check_missing_dist(idx):
            return None
        else:
            return self.dist[idx].ppf(q) 

    def cdf(self, x, idx=0):
        """
        Aggregate loss distribution cumulative distribution function.

        :param x: quantiles where the cumulative distribution function is evaluated.
        :type x: ``float`` or ``int`` or ``numpy.ndarray``
        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        if self._check_missing_dist(idx):
            return None
        else:
            return self.dist[idx].cdf(x)

    def sf(self, x, idx=0):
        """
        Aggregate loss distribution survival function.

        :param x: quantiles where the survival functionis evaluated.
        :type x: ``float`` or ``int`` or ``numpy.ndarray``
        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: survival function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        if self._check_missing_dist(idx):
            return None
        else:
            return self.dist[idx].sf(x)

    def rvs(self, size=1, random_state=None, idx=0):
        """
        Random variates generator function.

        :param size: random variates sample size, optional (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator, optional (no default).
        :type random_state: ``int``
        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``
        """

        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        if self._check_missing_dist(idx):
            return None
        else:
            return self.dist[idx].rvs(size, random_state)

    def mean(self, idx=0, use_dist=True):
        """
        Mean of the aggregate loss.

        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :param use_dist: If True, the mean is calculated from the (approximated) aggregate loss distributon.
                     If False, the mean is computed from the underlying frequency-severity loss model.
                     The latter is possible only if there are no aggregate conditions in the layer of interest. 
        :type use_dist: ``bool``
        :return: mean of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        hf.assert_type_value(
            use_dist, 'use_dist', logger, bool
            )
        share = self.policystructure.layers[idx].share
        if use_dist:
            if self._check_missing_dist(idx):
                return None
            else:
                return self.dist[idx].mean() * share
        else:
            if self.policystructure.layers[idx]._check_presence_aggr_conditions():
                message = 'Aggregate conditions are present for Layer %d, use_dist=False not available.' % (idx+1)
                logger.warning(message)
                return None
            ded = self.policystructure.layers[idx].deductible
            cover = self.policystructure.layers[idx].cover
            # adjustment factor frequency model from model threshold to the 0 (i.e. reporting threshold) 
            # where the severity model also refers to.
            # 1 := self.severity.model.sf(0)
            # remark: since it is a censored moment it always starts at 0, not the deductible.
            factor = 1 / self.severity.model.sf(self.frequency.threshold)
            return self.frequency.model.mean() * factor * share * \
                self.severity.censored_mean(deductible=ded, cover=cover)

    def var(self, idx=0, use_dist=True):
        """
        Variance of the aggregate loss.

        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :param use_dist: If True, the mean is calculated from the (approximated) aggregate loss distributon.
                     If False, the mean is computed from the underlying frequency-severity loss model.
                     The latter is possible only if there are no aggregate conditions in the layer of interest. 
        :type use_dist: ``bool``
        :return: variance of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        hf.assert_type_value(
            use_dist, 'use_dist', logger, bool
            )
        if use_dist:
            if self._check_missing_dist(idx):
                return None
            else:
                return self.dist[idx].var()
        else:
            if self.policystructure.layers[idx]._check_presence_aggr_conditions():
                message = 'Aggregate conditions are present for Layer %d, use_dist=False not available.' % (idx+1)
                logger.warning(message)
                return None
            ded = self.policystructure.layers[idx].deductible
            cover = self.policystructure.layers[idx].cover
            freq = self.frequency.model
            sev = self.severity
            share = self.policystructure.layers[idx].share
            # adjustment factor frequency model from model threshold to the 0 (i.e. reporting threshold) 
            # where the severity model also refers to.
            # 1 := sev.model.sf(0)
            # remark: since it is a censored moment it always starts at 0, not the deductible.
            factor = 1 / sev.model.sf(self.frequency.threshold)
            if factor > 1:
                freq.par_deductible_reverter(1/factor)

            output1 = freq.mean() * sev.censored_var(deductible=ded, cover=cover)
            output2 = freq.var() * sev.censored_mean(deductible=ded, cover=cover)**2
            # restore original unadjusted frequency model
            if factor > 1:
                freq.par_deductible_adjuster(1/factor)

            return share * (output1 + output2)

    def std(self, idx=0, use_dist=True):
        """
        Standard deviation of the aggregate loss.

        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :param use_dist: If True, the mean is calculated from the (approximated) aggregate loss distributon.
                     If False, the mean is computed from the underlying frequency-severity loss model.
                     The latter is possible only if there are no aggregate conditions in the layer of interest. 
        :type use_dist: ``bool``
        :return: standard deviation of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        return self.var(idx, use_dist)**(1/2)

    def skewness(self, idx=0, use_dist=True):
        """
        Skewness of the aggregate loss.

        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :param use_dist: If True, the skewness is calculated from the (approximated) aggregate loss distributon.
                     If False, the skewness is computed from the underlying frequency-severity loss model.
                     The latter is possible only if there are no aggregate conditions in the layer of interest. 
        :type use_dist: ``bool``
        :return: skewness of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        hf.assert_type_value(
            use_dist, 'use_dist', logger, bool
            )
        if use_dist:
            if self._check_missing_dist(idx):
                return None
            else:
                return self.dist[idx].skewness()
        else:
            if self.policystructure.layers[idx]._check_presence_aggr_conditions():
                message = 'Aggregate conditions are present for Layer %d, use_dist=False not available.' % (idx+1)
                logger.warning(message)
                return None
            freq = self.frequency.model
            sev = self.severity
            ded = self.policystructure.layers[idx].deductible
            cover = self.policystructure.layers[idx].cover
            sev_mean = sev.censored_mean(deductible=ded, cover=cover)
            sev_var = sev.censored_var(deductible=ded, cover=cover)
            sev_std = sev.censored_std(deductible=ded, cover=cover)
            sev_skew = sev.censored_skewness(deductible=ded, cover=cover)

            # adjustment factor frequency model from model threshold to the 0 (i.e. reporting threshold) 
            # where the severity model also refers to.
            # 1 := sev.model.sf(0)
            # remark: since it is a censored moment it always starts at 0, not the deductible.
            factor = 1 / sev.model.sf(self.frequency.threshold)
            if factor > 1:
                freq.par_deductible_reverter(1/factor)

            den1 = freq.mean() * sev.censored_var(deductible=ded, cover=cover)
            den2 = freq.var() * sev.censored_mean(deductible=ded, cover=cover)**2

            num1 = freq.skewness() * freq.std()**3 * sev_mean**3
            num2 = 3 * freq.var() * sev_mean * sev_var
            num3 = freq.mean() * sev_skew * sev_std**3

            # restore original unadjusted frequency model
            if factor > 1:
                freq.par_deductible_adjuster(1/factor)

            return (num1 + num2 + num3) / (den1 + den2)**(3/2)

    def coeff_variation(self, idx=0, use_dist=True):
        """
        Coefficient of variation (CoV) of the aggregate loss.

        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :param use_dist: If True, the CoV is calculated from the (approximated) aggregate loss distributon.
                     If False, the CoV is computed from the underlying frequency-severity loss model.
                     The latter is possible only if there are no aggregate conditions in the layer of interest. 
        :type use_dist: ``bool``
        :return: CoV of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        hf.assert_type_value(
            use_dist, 'use_dist', logger, bool
            )
        if use_dist:
            if self._check_missing_dist(idx):
                return None
            else:
                return self.dist[idx].std() / self.dist[idx].mean() 
        else:
            if self.policystructure.layers[idx]._check_presence_aggr_conditions():
                message = 'Aggregate conditions are present for Layer %d, use_dist=False not available.' % (idx+1)
                logger.warning(message)
                return None
            ded = self.policystructure.layers[idx].deductible
            cover = self.policystructure.layers[idx].cover
            freq = self.frequency.model
            sev = self.severity
            # adjustment factor frequency model from model threshold to the 0 (i.e. reporting threshold) 
            # where the severity model also refers to.
            # 1 := sev.model.sf(0)
            # remark: since it is a censored moment it always starts at 0, not the deductible.
            factor = 1 / sev.model.sf(self.frequency.threshold)
            if factor > 1:
                freq.par_deductible_reverter(1/factor)

            output1 = freq.mean() * sev.censored_var(deductible=ded, cover=cover)
            output2 = freq.var() * sev.censored_mean(deductible=ded, cover=cover)**2
            std = (output1 + output2)**(1/2)
            mean = freq.mean() * sev.censored_mean(deductible=ded, cover=cover)
            # restore original unadjusted frequency model
            if factor > 1:
                freq.par_deductible_adjuster(1/factor)

            return std / mean

    def _reinstatements_costing_adjuster(
        self,
        dist,
        aggr_deductible,
        n_reinst,
        cover,
        reinst_percentage
        ):
        """
        Reinstatements costing premium adjustment. Multiplicative factor.

        :param dist: aggregate loss distribution (before aggregate conditions).
        :type dist: ``dict``
        :param aggr_deductible: aggregate deductible.
        :type aggr_deductible: ``int`` or ``float``
        :param n_reinst: Number of reinstatements.
        :type n_reinst: ``int``
        :param cover: cover.
        :type cover: ``int`` or ``float``
        :param reinst_percentage: percentage of reinstatements layers (default value is 0), typically a value in [0, 1].
        :type reinst_percentage: ``int`` or ``float`` or ``np.array``
        :return: reinstatements costing adjustment.
        :rtype: ``float`` or ``numpy.ndarray``
        """

        output = 1
        if np.any(reinst_percentage) > 0:
            lower_k = np.arange(start=0, stop=n_reinst)
            dlk = self._stop_loss_costing(
                dist=dist,
                cover=cover,
                deductible=aggr_deductible + lower_k * cover
                )
            den = 1 + np.sum(dlk * reinst_percentage) / cover
            output = output / den
        return output

    def _stop_loss_costing(self, dist, cover, deductible):
        """
        Stop loss costing via stop loss transformation.
        Compute the expected value of layer transformed aggregate loss distribution.

        :param dist: aggregate loss distribution (before aggregate conditions).
        :type dist: ``dict``
        :param cover: cover.
        :type cover: ``int`` or ``float`` or ``numpy.ndarray``
        :param deductible: deductible.
        :type deductible: ``int`` or ``float`` or ``numpy.ndarray``
        :return: expected value of layer transformed aggregate loss distribution.
        :rtype: ``numpy.ndarray``
        """
        cover = np.asarray(cover).ravel()
        deductible = np.asarray(deductible).ravel()
        output = np.sum(
            hf.layerFunc(nodes=dist.nodes, cover=cover, deductible=deductible) * dist.pmf,
            axis=1
            )
        return output

    def costing(self):
        """
        Actuarial costing (also referred to as risk costing) of (re)insurance covers,
        such as quota share, excess-of-loss (including reinstatements or aggregate conditions) and stop loss.

        :return: Void
        :rtype: ``None``
        """

        pure_premiums = [None] * self.policystructure.length
        pure_premiums_dist = [None] * self.policystructure.length
        for idx in range(self.policystructure.length):
            hf.assert_type_value(
                idx, 'idx', logger, int,
                upper_bound=self.policystructure.length,
                lower_bound=0,
                upper_close=False
                )
            layer = self.policystructure.layers[idx]

            if layer._check_presence_aggr_conditions() and self.dist[idx] is None:
                # if there are aggregate conditions, approximated aggregate loss distribution is required.
                message = 'Layer %d: costing is omitted as aggr_loss_dist_method is missing' %(idx+1)
                logger.warning(message)
                continue 

            if self.dist[idx] is not None:
                # if approximated aggregate loss distribution is available use it to estimate the premium.
                # need to reintroduce the share for costing and apply it later.
                dist_excl_aggr_cond = self.__dist_excl_aggr_cond[idx]
                dist_excl_aggr_cond.nodes = dist_excl_aggr_cond.nodes / layer.share

                if layer.category in {'xlrs', 'xl/sl'}:
                    premium = self._stop_loss_costing(
                            dist=dist_excl_aggr_cond, # self.dist[idx],
                            cover=layer.aggr_cover,
                            deductible=layer.aggr_deductible
                            )
                # adjustment only if xl with reinstatements
                if layer.category in {'xlrs'}:
                    premium *= self._reinstatements_costing_adjuster(
                            dist=dist_excl_aggr_cond,
                            aggr_deductible=layer.aggr_deductible,
                            n_reinst=layer.n_reinst,
                            cover=layer.cover,
                            reinst_percentage=layer.reinst_percentage
                            )

                # re apply the partecipation share        
                premium *= layer.share
                pure_premiums_dist[idx] = premium.item()
            
            if not layer._check_presence_aggr_conditions():
                # if there are no aggregate conditions,
                # calculate 'exact' premium without approximated
                # aggrgeate loss distribution.
                pure_premiums[idx] = self.mean(idx=idx, use_dist=False).item()

        self.__pure_premium_dist = pure_premiums_dist
        self.__pure_premium = pure_premiums 
        return  

    def print_costing_specs(self, idx=0):
        """
        Print costing information of a given layer (specified via its index).
        
        :param idx: index corresponding to the policystructure layer of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: Void
        :rtype: ``None``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=self.policystructure.length,
            lower_bound=0,
            upper_close=False
            )

        if self.pure_premium_dist[0] is None:
            logger.info('Execution skipped, run costing before.')
            return

        layer = self.policystructure.layers[idx]
        premium = self.pure_premium[idx]
        premium_dist = self.pure_premium_dist[idx]
     
        data = [
            ['Cover', layer.cover],
            ['Deductible', layer.deductible]
        ]
        if layer.category == 'xlrs':
            data.extend([['Reinstatements (no.)', layer.n_reinst]])
            if isinstance(layer.reinst_percentage, (list, np.ndarray)):
                i = 1
                for percentage in layer.reinst_percentage:
                    data.extend([['Reinst. layer percentage ' + str(i), percentage]])
                    i = i + 1
            else:
                data.extend([['Reinst. layer percentage', layer.reinst_percentage]])
        elif layer.category == 'xl/sl':
            data.extend([['Aggregate cover', layer.aggr_cover]])

        data.extend([['Aggregate deductible', layer.aggr_deductible]])
        data.extend([['Pure premium (dist est.) before share partecip.', round(premium_dist / layer.share, 2)]])
        if premium is not None:
            data.extend([['Pure premium before share partecip.', round(premium / layer.share, 2)]])
        data.extend([['Share partecip.',  layer.share]])
        data.extend([['Pure premium (dist est.)', round(premium_dist, 2)]])
        if premium is not None:
            data.extend([['Pure premium', round(premium, 2)]])

        print('{: >40} {: >25} '.format(' ', *['Costing Summary: Layer ' + str(idx+1)]))
        print('{: >10} {: >70}'.format(' ', *['==========================================================================']))
        print('{: >10} {: >55} {: >15} '.format(' ', *['Quantity', 'Value']))
        print('{: >10} {: >70}'.format(' ', *['==========================================================================']))
        for row in data:
            print('{: >10} {: >55} {: >15}'.format(' ', *row))
        return

    def print_aggr_loss_method_specs(self, idx=0):
        """
        Print information of the aggregate loss distribution approximation for a given layer (specified via its index).

        :param idx: index corresponding to the policystructure layer of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: Void
        :rtype: None
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=self.policystructure.length,
            lower_bound=0,
            upper_close=False
            )

        if self.aggr_loss_dist_method is None:
            logger.info('Execution skipped, set aggr_loss_method before.')
            return

        data = [
            ['Aggregate loss dist. method', self.aggr_loss_dist_method]
        ]

        if self.aggr_loss_dist_method == 'mc':
            data.extend([
                ['Number of simulation', self.n_sim],
                ['Random state', self.random_state],
            ])
        else:
            n_sev_to_print = self.n_sev_discr_nodes
            # if self.policystructure.layers[idx].cover < float('inf'):
            #     n_sev_to_print += 1

            data.extend([
                ['n_aggr_dist_nodes', self.n_aggr_dist_nodes],
                ['Sev. discr. method', self.sev_discr_method],
                ['Sev. discr. step', self.sev_discr_step],
                ['Number of sev. discr. nodes', n_sev_to_print]
            ])
            if self.aggr_loss_dist_method == 'fft':
                data.extend([
                    ['Tilt flag', self.tilt],
                    ['Tilt parameter', self.tilt_value],
                ])

        print('{: >10} {: >35} '.format(' ', *['Aggregate Loss Distribution: layer ' + str(idx+1)]))
        print('{: >10} {: >50}'.format(' ', *['======================================================']))
        print('{: >10} {: >35} {: >15} '.format(' ', *['Quantity', 'Value']))
        print('{: >10} {: >50}'.format(' ', *['======================================================']))
        for row in data:
            print('{: >10} {: >35} {: >15}'.format(' ', *row))
        return

    def print_policy_layer_specs(self, idx=0):
        """
        Print policy structure information of a given layer (specified via its index).

        :param idx: index corresponding to the policystructure layer of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: Void
        :rtype: None
        """

        layer = self.policystructure.layers[idx]

        data = [
            ['Deductible', layer.deductible],
            ['Cover', layer.cover],
            ['Aggregate deductible', layer.aggr_deductible],
        ]
        if layer.category == 'xlrs':
            data.extend([['Reinstatements (no.)', layer.n_reinst]])
            if isinstance(layer.reinst_percentage, (list, np.ndarray)):
                i = 1
                for percentage in layer.reinst_percentage:
                    data.extend([['Reinst. layer percentage ' + str(i), percentage]])
                    i = i + 1
            else:
                data.extend([['Reinst. layer percentage', layer.reinst_percentage]])
        elif layer.category == 'xl/sl':
            data.extend([['Aggregate cover', layer.aggr_cover]])
        data.extend([['Share partecipation',  layer.share]])

        print('{: >10} {: >35} '.format(' ', *['Policy Structure Summary: layer ' + str(idx+1)]))
        print('{: >10} {: >40}'.format(' ', *['============================================']))
        print('{: >10} {: >25} {: >15} '.format(' ', *['Specification', 'Value']))
        print('{: >10} {: >40}'.format(' ', *['============================================']))
        for row in data:
            print('{: >10} {: >25} {: >15}'.format(' ', *row))
        return

    def _check_missing_dist(self, idx=0):
        """
        Check that the aggregate loss distribution is missing.
        Helper method called before executing other methods based on ``dist`` property.

        :param idx: index corresponding to the policystructure layer of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: True or False, depending whether ``dist`` is missing (None) or not.
        :rtype: ``bool``
        """
        output = hf.check_none(
            [self.dist[idx]],
            logger,
            'warning',
            message='Execution stopped. Missing ``dist``. Please execute ``dist_calculate`` first.'
            )
        return output

    def plot_dist_cdf(self, idx=0, log_x_scale=False, log_y_scale=False, **kwargs):
        """
        Plot the cumulative distribution function of the aggregate loss distribution.

        :param idx: index corresponding to the policystructure layer of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
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

        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist) - 1,
            lower_bound=0
        )

        if self._check_missing_dist(idx):
            return None
        
        hf.assert_type_value(log_x_scale, 'log_x_scale', logger, bool)
        hf.assert_type_value(log_y_scale, 'log_y_scale', logger, bool)

        x_ = self.dist[idx].nodes
        y_ = self.dist[idx].cumprobs

        figure = plt.figure()
        ax = figure.add_subplot(111)

        ax.step(x_, y_, '-', where='post', **kwargs)
        if log_y_scale:
            ax.set_yscale('log')
        if log_x_scale:
            ax.set_xscale('log')
        ax.set_title('Aggregate loss cumulative distribution function')
        ax.set_ylabel('cdf')
        ax.set_xlabel('nodes')
        return ax