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
                    When reinstatements are free (loadings = 0), an alternative parametrization is aggregate cover = (number of reinstatement + 1) * cover.
                    E.g. When the number of reinstatements = 0, the aggregate cover is equal to the cover,
                    when number of reinstatements is infinity there is no aggregate cover
                    (the aggregate cover is infinity).
    :type n_reinst: ``int``
    :param reinst_loading: loadings of reinstatements layers (default value is 0), typically a value in [0, 1].
    :type reinst_loading: ``int`` or ``float`` or ``np.array``
    :param maintenance_deductible: maintenance deductible, sometimes referred to as residual each-and-every-loss deductible (default is 0). Non-zero maintenance deductible applies to retention layers only.
    :type maintenance_deductible: ``int`` or ``float``
    :param share: Partecipation share of the layer (default is 1).
    :type share: ``float``
    :param basis: layer basis (default is 'regular'). One of 'regular', 'drop-down', 'stretch-down'. 
    :type basis: ``string``
    """

    def __init__(
        self,
        cover=float('inf'),
        deductible=0,
        aggr_cover=float('inf'),
        aggr_deductible=0,
        n_reinst=float('inf'),
        reinst_loading=0,
        maintenance_deductible=0,
        share=1,
        basis='regular',
        ):

        self.cover = cover
        self.deductible = deductible
        self.aggr_deductible = aggr_deductible
        self.aggr_cover = aggr_cover
        self.n_reinst = n_reinst
        self.reinst_loading = reinst_loading
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
    def reinst_loading(self):
        return self.__reinst_loading

    @reinst_loading.setter
    def reinst_loading(self, value):
        name = 'reinst_loading'
        name_two = 'reinst_loading size'

        if value is not None:
            if self.n_reinst == 0 or not self.n_reinst < float('inf'):
                # logger.info('reinst_loading set to None.')
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
        self.__reinst_loading = value

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
            self.reinst_loading
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
            'aggr_cover', 'n_reinst', 'reinst_loading',
            'maintenance_deductible', 'share',
            'basis'
            }
    
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
            # logger.info('n_reinst has been provided. aggr_cover set accordingly for internal consistency')
            self.aggr_cover = self.cover * (self.n_reinst + 1) # float('inf') # self.cover * (self.n_reinst + 1)
            self.__category = 'xlrs'
        else: # non 'xlrs' cases
            self.n_reinst = None
            self.reinst_loading = None
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
    """

    def __init__(self, dist, par):
        self.dist = dist
        self.par = par
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
 
    def stop_loss_transformation(self, cover, deductible, size=50000):
        """
        Approximated stop loss transformation function.
        General method for severity class, overridden by distribution specific implementation if available.
        
        :param cover: cover, also referred to as limit. cover plus deductible is the upper priority or severity 'exit point'.
        :type cover: ``int``, ``float``
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int``, ``float``
        :param size: inner random variates sample size to approximate the integral (default is 50000).
        :type size: ``int``, optional
        :return: stop_loss_transformation value.
        :rtype: ``numpy.float``
        """

        hf.assert_type_value(deductible, 'deductible', logger, type=(int, float),
        lower_bound=0, upper_bound=float('inf'), upper_close=False)
        hf.assert_type_value(cover, 'cover', logger, type=(int, float), lower_bound=0, lower_close=False)
        hf.assert_type_value(size, 'size', logger, type=(int, float), lower_bound=1, lower_close=False)
        size = int(size)

        t_ = np.linspace(start=deductible, stop=deductible+cover, num=size, endpoint=True)
        output = np.sum(self.model.sf(t_))

        return output

    def discretize(
        self,
        discr_method,
        n_discr_nodes,
        discr_step,
        cover,
        deductible
        ):
        """
        Severity discretization according to the discretization method selected by the user.

        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int`` or ``float``
        :param cover: cover, also referred to as limit.
        :type cover: ``int`` or ``float``
        :param discr_method: severity discretization method. One of 'massdispersal', 'localmoments'.
        :type discr_method: ``str``
        :param discr_step: severity discretization step.
        :type discr_step: ``float``
        :param n_discr_nodes: number of nodes of the discretized severity.
        :type n_discr_nodes: ``int``
        :return: discrete severity, nodes sequence and discrete probabilities.
        :rtype: ``dict``
        """
        
        hf.assert_member(discr_method, config.SEV_DISCRETIZATION_METHOD, logger)
        hf.assert_type_value(deductible, 'deductible', logger, type=(int, float), lower_bound=0)
        hf.assert_type_value(cover, 'cover', logger, type=(int, float), lower_bound=0, lower_close=False)
        exit_point = cover + deductible
        hf.assert_type_value(n_discr_nodes, 'n_discr_nodes', logger, type=(int, float), lower_bound=1)
        n_discr_nodes = int(n_discr_nodes)

        if exit_point < float('inf'):
            discr_step = cover / (n_discr_nodes)
            n_discr_nodes = n_discr_nodes - 1
            # discr_step = cover / n_discr_nodes
            # logger.info('discr_step set to %s.' %(discr_step))

        hf.assert_type_value(discr_step, 'discr_step', logger, type=(int, float), lower_bound=0, lower_close=False)
        discr_step = float(discr_step)
        
        if discr_method == 'massdispersal':
            return Calculator.mass_dispersal(
                self,
                deductible,
                exit_point,
                discr_step,
                n_discr_nodes
                )
        elif discr_method == 'localmoments':
            return Calculator.local_moments(
                self,
                deductible,
                exit_point,
                discr_step,
                n_discr_nodes
                )

    def plot_discretized_severity(self,
                                  discr_method,
                                  n_discr_nodes,
                                  discr_step,
                                  cover,
                                  deductible,
                                  *args):



        sevdict = self.discretize(
            discr_method=discr_method,
            n_discr_nodes=n_discr_nodes,
            discr_step=discr_step,
            cover=cover,
            deductible=deductible
        )

        n_discr_nodes = int(n_discr_nodes)
        exit_point = cover + deductible
        if exit_point < float('inf'):
            discr_step = cover / (n_discr_nodes)

        discr_step = float(discr_step)

        x_ = np.concatenate([np.array([sevdict['nodes'][0]-discr_step]), sevdict['nodes']])
        y_ = np.concatenate([np.zeros(1,), np.cumsum(sevdict['fj'])])
        plt.step(x_, y_, '-', where='pre', *args)
        plt.title('Discretized severity cumulative distribution function')
        plt.xlabel('cdf')
        plt.ylabel('nodes')
        plt.show()

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
    :type random_state: ``int``, optional
    :param n_aggr_dist_nodes: number of nodes in the approximated aggregate loss distribution.
    :type n_aggr_dist_nodes: ``int``
    :param sev_discr_method: severity discretization method. One of 'massdispersal', 'localmoments'.
    :type sev_discr_method: ``str``
    :param sev_discr_step: severity discretization step.
    :type sev_discr_step: ``float``
    :param n_sev_discr_nodes: number of nodes of the discretized severity.
    :type n_sev_discr_nodes: ``int``
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
        self.__dist = None
        self.__dist_excl_aggr_cond = None
        self.__pure_premium = None
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
            logger.info('Aggregate loss distribution calculation is omitted as aggr_loss_dist_method is missing')
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
            
            # adjust original frequency model for the calculation.
            self.frequency.model.par_deductible_adjuster(
                1 - self.severity.model.cdf(layer.deductible)
                )

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
                hf.assert_not_none(
                    value=self.sev_discr_step,
                    name='sev_discr_step',
                    logger=logger
                )
                hf.assert_not_none(
                    value=self.n_sev_discr_nodes,
                    name='n_sev_discr_nodes',
                    logger=logger
                )
                hf.assert_not_none(
                    value=self.n_aggr_dist_nodes,
                    name='n_aggr_dist_nodes',
                    logger=logger
                    )
                hf.check_condition(
                    value=self.n_aggr_dist_nodes,
                    check=self.n_sev_discr_nodes,
                    name='n_aggr_dist_nodes',
                    logger=logger,
                    type='>='
                )

                # if self.n_sev_discr_nodes/self.n_aggr_dist_nodes >= 0.90:
                #     logger.warning(
                #         'Setting similar values for n_aggr_dist_nodes and n_sev_discr_nodes ' +
                #         'might compromise the quality of the aggregate loss distribution approximation.'
                #     )

                sevdict = self.severity.discretize(
                    discr_method=self.sev_discr_method,
                    n_discr_nodes=self.n_sev_discr_nodes,
                    discr_step=self.sev_discr_step,
                    cover=layer.cover,
                    deductible=layer.deductible
                    )
                
                # adjustment of the discr_step to use in the aggregate distribution calculation below.
                # severity.discretize() method performs it within its body.
                if layer.cover < float('inf'):
                    self.sev_discr_step = (layer.cover) / self.n_sev_discr_nodes
                    # self.n_sev_discr_nodes = self.n_sev_discr_nodes - 1

                if self.aggr_loss_dist_method == 'recursion':

                    logger.info('Approximating aggregate loss distribution via Panjer recursion')
                    aggr_dist_excl_aggr_cond = Calculator.panjer_recursion(
                        severity=sevdict,
                        discr_step=self.sev_discr_step,
                        n_aggr_dist_nodes=self.n_aggr_dist_nodes,
                        # n_sev_discr_nodes=self.n_sev_discr_nodes,
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
                        n_aggr_dist_nodes=self.n_aggr_dist_nodes,
                        # n_sev_discr_nodes=self.n_sev_discr_nodes 
                        )
                    logger.info('FFT completed') 

            # restore original unadjusted frequency model
            self.frequency.model.par_deductible_reverter(
                1 - self.severity.model.cdf(layer.deductible)
            )

            aggr_dist_list_excl_aggr_cond[i] = distributions.PWC(
                    nodes=aggr_dist_excl_aggr_cond['nodes'],
                    cumprobs=aggr_dist_excl_aggr_cond['cdf']
                )

            aggr_dist_incl_aggr_cond = self._apply_aggr_conditions(
                dist=aggr_dist_excl_aggr_cond,
                deductible=layer.aggr_deductible,
                cover=layer.aggr_cover
                )
            aggr_dist_list_incl_aggr_cond[i] = distributions.PWC(
                    nodes=aggr_dist_incl_aggr_cond['nodes'], #inodes,
                    cumprobs=aggr_dist_incl_aggr_cond['cdf'] # icumprobs
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
        Aggregate loss distribution moment of order n.

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
        self._check_dist(idx)
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
        self._check_dist(idx)
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
        self._check_dist(idx)
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
        self._check_dist(idx)
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
        self._check_dist(idx)
        return self.dist[idx].rvs(size, random_state)

    def mean(self, idx=0):
        """
        Mean of the aggregate loss.

        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: mean of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        self._check_dist(idx)
        return self.dist[idx].mean()

    def std(self, idx=0):
        """
        Standard deviation of the aggregate loss.

        :param n: list index corresponding to the layer loss distribution of interest (default is 0).
                  See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type n: ``idx``
        :return: standard deviation of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        self._check_dist(idx)
        return self.dist[idx].std()

    def skewness(self, idx=0):
        """
        Skewness of the aggregate loss.

        :param idx: list index corresponding to the layer loss distribution of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: skewness of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist)-1,
            lower_bound=0
            )
        self._check_dist(idx)
        return self.dist[idx].skewness()

    def _reinstatements_costing_adjuster(
        self,
        dist,
        aggr_deductible,
        n_reinst,
        cover,
        reinst_loading
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
        :param reinst_loading: loadings of reinstatements layers (default value is 0), typically a value in [0, 1].
        :type reinst_loading: ``int`` or ``float`` or ``np.array``
        :return: reinstatements costing adjustment.
        :rtype: ``float`` or ``numpy.ndarray``
        """

        output = 1
        if np.any(reinst_loading) > 0:
            lower_k = np.arange(start=0, stop=n_reinst)
            dlk = self._stop_loss_costing(
                dist=dist,
                cover=cover,
                deductible=aggr_deductible + lower_k * cover
                )
            den = 1 + np.sum(dlk * reinst_loading) / cover
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
        if (self.dist is None):
            logger.info('Costing is omitted as aggr_loss_dist_method is missing')
            return 

        pure_premiums = [None] * self.policystructure.length
        for idx in range(self.policystructure.length):
            hf.assert_type_value(
                idx, 'idx', logger, int,
                upper_bound=self.policystructure.length,
                lower_bound=0,
                upper_close=False
                )
            self._check_dist(idx)
            layer = self.policystructure.layers[idx]
            dist_excl_aggr_cond = self.__dist_excl_aggr_cond[idx]

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
                        reinst_loading=layer.reinst_loading
                        )
                            
            premium *= layer.share
            pure_premiums[idx] = premium.item()
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

        if self.pure_premium is None:
            logger.info('Execution skipped, run costing before.')
            return

        layer = self.policystructure.layers[idx]
        premium = self.pure_premium[idx]
     
        data = [
            ['Cover', layer.cover],
            ['Deductible', layer.deductible]
        ]
        if layer.category == 'xlrs':
            data.extend([['Reinstatements (no.)', layer.n_reinst]])
            if isinstance(layer.reinst_loading, (list, np.ndarray)):
                i = 1
                for loading in layer.reinst_loading:
                    data.extend([['Reinst. layer loading ' + str(i), loading]])
                    i = i + 1
            else:
                data.extend([['Reinst. layer loading', layer.reinst_loading]])
        elif layer.category == 'xl/sl':
            data.extend([['Aggregate cover', layer.aggr_cover]])

        data.extend([['Aggregate deductible', layer.aggr_deductible]])
        data.extend([['Pure premium before share partecip.', round(premium/layer.share, 2)]])
        data.extend([['Share partecip.',  layer.share]])
        data.extend([['Pure premium', round(premium, 2)]])

        print('{: >20} {: >25} '.format(' ', *['Costing Summary: Layer ' + str(idx)]))
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
        print('{: >10} {: >35} {: >15} '.format(' ', *['Quantity', 'Value']))
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
        for row in data:
            print('{: >10} {: >35} {: >15}'.format(' ', *row))
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
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
        print('{: >10} {: >35} {: >15} '.format(' ', *['Quantity', 'Value']))
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
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
            if isinstance(layer.reinst_loading, (list, np.ndarray)):
                i = 1
                for loading in layer.reinst_loading:
                    data.extend([['Reinst. layer loading ' + str(i), loading]])
                    i = i + 1
            else:
                data.extend([['Reinst. layer loading', layer.reinst_loading]])
        elif layer.category == 'xl/sl':
            data.extend([['Aggregate cover', layer.aggr_cover]])
        data.extend([['Share portion',  layer.share]])

        print('{: >10} {: >35} '.format(' ', *['Policy Structure Summary: layer ' + str(idx+1)]))
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
        print('{: >10} {: >35} {: >15} '.format(' ', *['Specification', 'Value']))
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
        for row in data:
            print('{: >10} {: >35} {: >15}'.format(' ', *row))
        return

    def _check_dist(self, idx=0):
        """
        Check that the aggregate loss distribution is not missing.
        Helper method called before executing other methods based on ``dist`` property.

        :param idx: index corresponding to the policystructure layer of interest (default is 0).
                    See 'index_to_layer_name' and 'layer_name_to_index' PolicyStructure methods.
        :type idx: ``int``
        :return: Void
        :rtype: None
        """
        hf.assert_not_none(
            value=self.dist[idx], name='dist', logger=logger
        )

    def plot_dist_cdf(self, idx=0, *args):

        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.dist) - 1,
            lower_bound=0
        )
        self._check_dist(idx)

        x_ = np.concatenate([np.array([self.dist[idx].nodes[0] - self.dist[idx].nodes[2]-self.dist[idx].nodes[1]]), self.dist[idx].nodes])

        y_ = self.dist[idx].cdf(x_)
        plt.step(x_, y_, '-', where='pre', *args)
        plt.title('Aggregate loss cumulative distribution function')
        plt.xlabel('cdf')
        plt.ylabel('nodes')
        plt.show()

        return self.dist[idx].mean()
