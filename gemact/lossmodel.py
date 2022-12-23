# from .libraries import *
# from . import config
# from . import helperfunctions as hf
# from . import distributions as distributions
# from .calculators import LossModelCalculator as Calculator 
from libraries import *
import config
import helperfunctions as hf
import distributions as distributions
from calculators import LossModelCalculator as Calculator 

quick_setup()
logger = log.name('lossmodel')

class PolicyStructure:
    """
    Policy structure component of a loss model.

    :param layers: Non-proportional layer (default is infinity-xs-0 layer).
    :type layers: ``Layer``, ``LayerTower``, ``list``
    :param share: Partecipation share of the layer (default is 1).
    :type share: ``float``
    """

    def __init__(self, layers=None, shares=None):
        self.layers = layers if layers is not None else Layer()
        self.shares = shares if shares is not None else 1.00
        self._check_share_and_layer()

    @property
    def layers(self):
        return self.__layers

    @layers.setter
    def layers(self, value):
        hf.assert_type_value(
            value, 'layers',
            type=(LayerTower, Layer, list), logger=logger
            )
        if isinstance(value, Layer):
            value = [value]
            logger.info('layers converted to list for internal consistency')
        elif isinstance(value, list):
            all(hf.assert_type_value(
                item, 'layers item', type=Layer, logger=logger
                ) for item in value)
            # value = LayerTower(*value)
        self.__layers = value
    
    @property
    def shares(self):
        return self.__shares

    @shares.setter
    def shares(self, value):

        if hasattr(value, '__len__'):
            hf.assert_type_value(value, 'share', logger, list)
            for val in value:
                hf.assert_type_value(
                    val, 'share', logger, (float, int),
                    upper_bound=1, upper_close=True,
                    lower_bound=0, lower_close=True
                )
        else:
            hf.assert_type_value(
                value, 'share', type=(float, int), logger=logger,
                upper_bound=1, upper_close=True,
                lower_bound=0, lower_close=True
                )
            value = [value]
        self.__shares = value

    @property
    def length(self):
        return len(self.layers)

    def _check_share_and_layer(self):
        """
        Check correctness and consistency between share and layer policy component.
        """
        if len(self.shares) != len(self.layers):
            hf.assert_equality(len(self.shares), 1, 'shares', logger)
            logger.info('Remark: shares value is applied to all layers items')
            self.shares = [self.shares] * len(self.layers)

class Layer:
    """
    Policy structure non-proportional layer.

    :param deductible: non-ranking each-and-every-loss deductible, also referred to as retention or priority (default value is 0).
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
        basis='regular',
        ):

        self.cover = cover
        self.deductible = deductible
        self.aggr_deductible = aggr_deductible
        self.aggr_cover = aggr_cover
        self.n_reinst = n_reinst
        self.reinst_loading = reinst_loading
        self.maintenance_deductible = maintenance_deductible
        self.basis = basis
        self._check_and_set_category()

    @property
    def name(self):
        output = ('%s-xs-%s, %s-xs-%s in aggr.' %(self.cover, self.deductible, self.aggr_cover, self.aggr_deductible))
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
                logger.info('reinst_loading set to None.')
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
                    hf.assert_equality(
                        value.shape[0], self.n_reinst,
                        name_two, logger
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
    def basis(self):
        return self.__basis

    @basis.setter
    def basis(self, value):
        hf.assert_member(value, config.POLICY_LAYER_BASIS, logger)
        self.__basis = value

    @property
    def category(self):
        return self.__category

    @property
    def exit_point(self):
        return self.deductible + self.cover
    
    @property
    def identifier(self):
        output = '{}_{}_{}_{}_{}_{}'.format(
            self.deductible,
            self.cover,
            self.aggr_deductible,
            self.aggr_cover,
            self.n_reinst,
            self.reinst_loading
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
            'aggr_cover', 'n_reinst', 'reinst_loading', 'exit_point',
            'maintenance_deductible', 'basis'
            }
    
    def _check_and_set_category(self):
        """
        Method that check and set the category of the layer.

        :return: Void.
        :rtype: ``None``
        """
        # xlrs case
        if self.n_reinst is not None and np.isfinite(self.n_reinst):
            hf.assert_not_equality(
                self.cover, float('inf'),
                'cover', logger
            )
            logger.info('n_reinst has been provided. aggr_cover set accordingly for internal consistency')
            self.aggr_cover = self.cover * (self.n_reinst + 1)
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

class LayerTower(list):
    """
    Policy structure tower of non-proportional layers.

    :param \\**args:
        See below

    :Keyword Arguments:
        * *args* (``Layers``) --
          Layer tower elements.
    """

    def __init__(self, *args):
        for arg in args:
            hf.assert_type_value(arg, 'item', logger, Layer)
        super(LayerTower, self).__init__(args)
        # self.extend(args)
        self.set_and_check_tower()

    def append(self, item):
        hf.assert_type_value(item, 'item', logger, Layer)
        super(LayerTower, self).append(item)
    
    def insert(self, item):
        hf.assert_type_value(item, 'item', logger, Layer)
        super(LayerTower, self).insert(item)
    
    def extend(self, *args):
        for arg in args:
            hf.assert_type_value(arg, 'item', logger, Layer)
        super(LayerTower, self).extend(args)
    
    def sort(self, key='deductible'):
        hf.assert_member(key, Layer.specs(), logger)
        super(LayerTower, self).sort(
            key=lambda x: getattr(x, key)
            )
    
    def remove_layer_loading(self):
        for elt in self:
            elt.reinst_loading = 0

    def set_and_check_tower(self):
        self.remove_duplicates()
        self.sort()
        hf.assert_equality(
            self[0].deductible, 0,
            logger=logger,
            name='First layer (retention) deductible'
            )
        for i in range(1, len(self)):
            hf.assert_not_equality(
                self[i].category,
                'xlrs',
                'category of ' + self[i].name,
                logger
            )
            hf.assert_equality(
                self[i].deductible,
                self[i-1].deductible + self[i-1].cover,
                'deductible of ' + self[i].name,
                logger
            )
            if self[i].basis == 'regular':
                logger.warning('Having regular basis may generate noncontiguous layers.')

    def remove_duplicates(self):
        memory = []
        for element in self:
            if element.identifier not in memory:
                memory.append(element.identifier)
            else:
                self.remove(element)
                logger.warning(
                    'Removing %s as a duplicate of another Layer.' %(element.name)
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
            n_discr_nodes = n_discr_nodes - 1
            discr_step = cover / (n_discr_nodes + 1)
            logger.info('discr_step set to cover/n_sev_discr_nodes.')

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

class LossModel:
    """
    Loss model for (re)insurance costing and risk modeling using a collective risk model framework.

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
        policystructure,
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
        self.__aggr_loss_dist = None
        self.__aggr_loss_dist_excl_aggr_cond = None
        self.__pure_premium = None
        self.aggr_loss_dist_calculate()
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
    def aggr_loss_dist(self):
        return self.__aggr_loss_dist
    
    @property
    def pure_premium(self):
        return self.__pure_premium

    def aggr_loss_dist_calculate(
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
        Approximation of aggregate loss distributions by calculating nodes, pdf, and cdf.
        Distributions can be accessed via the aggr_loss_dist property,
        which is a list of ``dict``, each one representing a aggregate loss distribution.

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
        :return: Void
        :rtype: ``None``
        """

        if (aggr_loss_dist_method is None) and (self.aggr_loss_dist_method is None):
            self.__aggr_loss_dist = [{'nodes': None, 'epmf': None, 'ecdf': None}]
            logger.info('Aggregate loss distribution calculation is omitted as aggr_loss_dist_method is missing')
            return
        
        for lay in self.policystructure.layers:
            lay._check_and_set_category()    

        aggr_dist_list_incl_aggr_cond = [None] * self.policystructure.length
        for i in range(self.policystructure.length):
            layer = self.policystructure.layers[i]
            
            # adjust original frequency model for the calculation.
            self.frequency.model.par_franchise_adjuster(
                1 - self.severity.model.cdf(layer.deductible)
                )

            if aggr_loss_dist_method is not None:
                self.aggr_loss_dist_method = aggr_loss_dist_method
            hf.assert_not_none(
                value=self.aggr_loss_dist_method,
                name='aggr_loss_dist_method',
                logger=logger
            )

            if n_aggr_dist_nodes is not None:
                self.n_aggr_dist_nodes = n_aggr_dist_nodes
            hf.assert_not_none(
                value=self.n_aggr_dist_nodes,
                name='n_aggr_dist_nodes',
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
                hf.assert_not_none(
                    value=self.random_state,
                    name='random_state',
                    logger=logger
                )
                
                logger.info('..Approximating aggregate loss distribution via Monte Carlo simulation..')
                aggr_dist_excl_aggr_cond = Calculator.mc_simulation(
                    severity=self.severity,
                    frequency=self.frequency,
                    cover=layer.cover,
                    deductible=layer.deductible,
                    n_sim=self.n_sim,
                    random_state=self.random_state
                    )
                logger.info('..MC simulation completed..')
                
            else:
                if sev_discr_method is not None:
                    self.sev_discr_method = sev_discr_method
                if sev_discr_step is not None:
                    self.sev_discr_step = sev_discr_step
                if n_sev_discr_nodes is not None:
                    self.n_sev_discr_nodes = n_sev_discr_nodes
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
                    logger.info('discr_step set to cover/n_sev_discr_nodes.') 

                if self.aggr_loss_dist_method == 'recursion':

                    logger.info('..Approximating aggregate loss distribution via Panjer recursion..')
                    aggr_dist_excl_aggr_cond = Calculator.panjer_recursion(
                        severity=sevdict,
                        discr_step=self.sev_discr_step,
                        n_aggr_dist_nodes=self.n_aggr_dist_nodes,
                        n_sev_discr_nodes=self.n_sev_discr_nodes,
                        frequency=self.frequency
                        )
                    logger.info('..Panjer recursion completed..')

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

                    logger.info('..Approximating aggregate loss distribution via FFT..')
                    aggr_dist_excl_aggr_cond = Calculator.fast_fourier_transform(
                        severity=sevdict,
                        discr_step=self.sev_discr_step,
                        tilt=self.tilt,
                        tilt_value=self.tilt_value,
                        frequency=self.frequency,
                        n_aggr_dist_nodes=self.n_aggr_dist_nodes,
                        n_sev_discr_nodes=self.n_sev_discr_nodes 
                        )
                    logger.info('..FFT completed..') 

            aggr_dist_incl_aggr_cond = self._apply_aggr_conditions(
                aggr_loss_dist=aggr_dist_excl_aggr_cond,
                deductible=layer.aggr_deductible,
                cover=layer.aggr_cover
                )

            aggr_dist_list_incl_aggr_cond[i] = aggr_dist_incl_aggr_cond
            
            # restore original unadjusted frequency model
            self.frequency.model.par_franchise_reverter(
                1 - self.severity.model.cdf(layer.deductible)
            )
            # go next i

        self.__aggr_loss_dist = aggr_dist_list_incl_aggr_cond
        return

    def _apply_aggr_conditions(self, aggr_loss_dist, cover, deductible):
        """
        Apply aggregate conditions, i.e. aggregate deductble and aggregate cover, to a aggregate loss distribution.

        :param aggr_loss_dist: aggregate loss distribution (before aggregate conditions).
        :type aggr_loss_dist: ``dict``
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
            nodes=aggr_loss_dist['nodes'],
            cover=cover,
            deductible=deductible
            ).ravel()
        
        output['nodes'] = np.unique(nodes)
        output['epmf'] = np.array(
            [aggr_loss_dist['epmf'][nodes == item].sum() for item in output['nodes']]
        )
        output['ecdf'] = np.cumsum(output['epmf'])
        return output

    def moment(self, central=False, order=1, idx=0):
        """
        Aggregate loss distribution moment.

        :param central: True if the moment is central, False if the moment is raw.
        :type central: ``bool``
        :param order: order of the moment, optional (default is 1).
        :type order: ``int``
        :param idx: list index corresponding to the loss distribution of interest (default is 0).
        :type idx: ``int``
        :return: approximated moment.
        :rtype: ``numpy.float64``
        """

        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.aggr_loss_dist)-1,
            lower_bound=0
            )
        self._check_aggr_loss_dist(idx)
        aggr_loss_dist_ = self.aggr_loss_dist[idx]

        hf.assert_type_value(central, 'central', logger, bool)
        hf.assert_type_value(order, 'order', logger, (int, float), lower_bound=0, lower_close=False)
        order = int(order)
        
        if self.aggr_loss_dist_method != 'mc':
            lmmean = np.sum(aggr_loss_dist_['epmf'] * aggr_loss_dist_['nodes'])
            return np.sum(aggr_loss_dist_['epmf'] * ((aggr_loss_dist_['nodes'] - (central * lmmean)) ** order))
        else:
            return np.mean((aggr_loss_dist_['nodes'] - central * np.mean(aggr_loss_dist_['nodes'])) ** order)

    def ppf(self, q, idx=0):
        """
        Aggregate loss distribution percent point function, a.k.a. the quantile function,
        inverse of the cumulative distribution function.

        :param q: probability.
        :type q: ``float`` or ``numpy.ndarray``
        :param idx: list index corresponding to the loss distribution of interest (default is 0).
        :type idx: ``int``
        :return: quantile.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        hf.assert_type_value(
            q, 'q', logger, (np.floating, int, float, list, np.ndarray)
            )
        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.aggr_loss_dist)-1,
            lower_bound=0
            )
        self._check_aggr_loss_dist(idx)
        aggr_loss_dist_ = self.aggr_loss_dist[idx]

        q = np.ravel(q)
        for item in q:
            hf.assert_type_value(item, 'q', logger, (float, int), upper_bound=1, lower_bound=0)
        
        y_ = np.append(0, aggr_loss_dist_['nodes'])
        z_ = np.append(0, aggr_loss_dist_['ecdf'])
        f = interp1d(z_, y_)
        return f(q)

    def cdf(self, x, idx=0):
        """
        Aggregate loss distribution cumulative distribution function.

        :param x: quantiles where the cumulative distribution function is evaluated.
        :type x: ``float`` or ``int`` or ``numpy.ndarray``
        :param idx: list index corresponding to the loss distribution of interest (default is 0).
        :type idx: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.aggr_loss_dist)-1,
            lower_bound=0
            )
        self._check_aggr_loss_dist(idx)
        aggr_loss_dist_ = self.aggr_loss_dist[idx]

        x = np.ravel(x)
        z_ = np.append(0, aggr_loss_dist_['ecdf'])
        y_ = np.append(0, aggr_loss_dist_['nodes'])
        f = interp1d(y_, z_)

        x[x <= 0] = 0
        x[x >= aggr_loss_dist_['nodes'][-1]] = aggr_loss_dist_['nodes'][-1]
        return f(x)

    def rvs(self, size=1, random_state=None, idx=0):
        """
        Random variates generator function.

        :param size: random variates sample size, optional (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator, optional (no default).
        :type random_state: ``int``
        :param idx: list index corresponding to the loss distribution of interest (default is 0).
        :type idx: ``int``
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``
        """

        hf.assert_type_value(
            idx, 'idx', logger, int,
            upper_bound=len(self.aggr_loss_dist)-1,
            lower_bound=0
            )
        self._check_aggr_loss_dist(idx)
        aggr_loss_dist_ = self.aggr_loss_dist[idx]

        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=0, lower_close=False)
        size = int(size)

        output = np.random.choice(aggr_loss_dist_['nodes'], size=size, p=aggr_loss_dist_['epmf'])
        return output

    def mean(self, idx=0):
        """
        Mean of the aggregate loss.

        :param idx: list index corresponding to the loss distribution of interest (default is 0).
        :type idx: ``int``
        :return: mean of the aggregate loss.
        :rtype: ``numpy.float64``
        """

        return self.moment(central=False, order=1, idx=idx)

    def std(self, idx=0):
        """
        Standard deviation of the aggregate loss.

        :param n: list index corresponding to the loss distribution of interest (default is 0).
        :type n: ``idx``
        :return: standard deviation of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        return self.moment(central=True, order=2, idx=idx) ** 1 / 2

    def skewness(self, idx=0):
        """
        Skewness of the aggregate loss.

        :param idx: list index corresponding to the loss distribution of interest (default is 0).
        :type idx: ``int``
        :return: skewness of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        return self.moment(central=True, order=3, idx=idx) / self.moment(central=True, order=2, idx=idx) ** 3 / 2

    def _reinstatements_costing_adjuster(
        self,
        aggr_loss_dist,
        aggr_deductible,
        n_reinst,
        cover,
        reinst_loading
        ):
        """
        Reinstatements costing premium adjustment. Multiplicative factor.

        :param aggr_loss_dist: aggregate loss distribution (before aggregate conditions).
        :type aggr_loss_dist: ``dict``
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
                cover=cover,
                deductible=aggr_deductible + lower_k * cover,
                aggr_loss_dist=aggr_loss_dist
                )
            den = 1 + np.sum(dlk * reinst_loading) / cover
            output = output / den
        return output

    def _stop_loss_costing(self, aggr_loss_dist, cover=None, deductible=None):
        """
        Stop loss costing via stop loss transformation.
        Compute the expected value of layer transformed aggregate loss distribution.

        :param aggr_loss_dist: aggregate loss distribution (before aggregate conditions).
        :type aggr_loss_dist: ``dict``
        :param cover: cover.
        :type cover: ``int`` or ``float`` or ``numpy.ndarray``
        :param deductible: deductible.
        :type deductible: ``int`` or ``float`` or ``numpy.ndarray``
        :return: expected value of layer transformed aggregate loss distribution.
        :rtype: ``numpy.ndarray``
        """
        if cover is None and deductible is None:
            output = np.average(
            a = aggr_loss_dist['nodes'],
            weights= aggr_loss_dist['epmf']
            )
        else:
            cover = np.asarray(cover).ravel()
            deductible = np.asarray(deductible).ravel()
            output = np.sum(
                hf.layerFunc(nodes=aggr_loss_dist['nodes'], cover=cover, deductible=deductible) * aggr_loss_dist['epmf'],
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
        if (self.__aggr_loss_dist[0]['nodes'] is None):
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
            self._check_aggr_loss_dist(idx)
            layer = self.policystructure.layers[idx]
            aggr_loss_dist_excl_aggr_cond = self.__aggr_loss_dist_excl_aggr_cond[idx]
            share_portion = self.policystructure.shares[idx]

            if layer.category in {'xlrs', 'xl/sl'}:
                premium = self._stop_loss_costing(
                        aggr_loss_dist = self.aggr_loss_dist[idx]
                        )
            # adjustment only if xl with reinstatements
            if layer.category in {'xlrs'}:
                premium *= self._reinstatements_costing_adjuster(
                    aggr_loss_dist=aggr_loss_dist_excl_aggr_cond,
                    aggr_deductible=layer.aggr_deductible,
                    n_resint=layer.n_reinst,
                    cover=layer.cover,
                    reinst_loading=layer.reinst_loading
                    ).item()
            
            premium *= share_portion
            pure_premiums[idx] = premium
        self.__pure_premium = pure_premiums 
        return  

    def print_costing_specs(self, idx=0):
        """
        Print costing information of a given layer (specified via its index).
        
        :param idx: index corresponding to the policystructure element of interest (default is 0).
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
            logger.info('..Computing costing..')
            self.costing()

        layer = self.policystructure.layers[idx]
        share_portion = self.policystructure.shares[idx]
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
        data.extend([['Pure premium before share portion', round(premium/share_portion, 2)]])
        data.extend([['Share portion',  share_portion]])
        data.extend([['Pure premium', round(premium, 2)]])

        print('{: >20} {: >25} '.format(' ', *['Costing Summary']))
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
        print('{: >10} {: >35} {: >15} '.format(' ', *['Quantity', 'Value']))
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
        for row in data:
            print('{: >10} {: >35} {: >15}'.format(' ', *row))
        return

    def print_aggr_loss_method_specs(self, idx=0):
        """
        Print information of the aggregate loss distribution approximation for a given layer (specified via its index).

        :param idx: index corresponding to the policystructure element of interest (default is 0).
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
            ['Aggregate loss dist. method', self.aggr_loss_dist_method],
            ['n_aggr_dist_nodes', self.n_aggr_dist_nodes],
        ]

        if self.aggr_loss_dist_method == 'mc':
            data.extend([
                ['Number of simulation', self.n_sim],
                ['Random state', self.random_state],
            ])
        else:
            n_sev_to_print = self.n_sev_discr_nodes
            if self.policystructure.layers[idx].cover < float('inf'):
                n_sev_to_print += 1

            data.extend([
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

        :param idx: index corresponding to the policystructure element of interest (default is 0).
        :type idx: ``int``
        :return: Void
        :rtype: None
        """

        layer = self.policystructure.layers[idx]
        share_portion = self.policystructure.shares[idx]

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
        data.extend([['Share portion',  share_portion]])

        print('{: >10} {: >35} '.format(' ', *['Policy Structure Summary: layer ' + str(idx+1)]))
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
        print('{: >10} {: >35} {: >15} '.format(' ', *['Specification', 'Value']))
        print('{: >10} {: >35}'.format(' ', *['====================================================================']))
        for row in data:
            print('{: >10} {: >35} {: >15}'.format(' ', *row))
        return

    def _check_aggr_loss_dist(self, idx=0):
        """
        Check that the aggregate loss distribution is not missing.
        Helper method called before executing other methods based on ``aggr_loss_dist`` property.

        :param idx: index corresponding to the policystructure element of interest (default is 0).
        :type idx: ``int``
        :return: Void
        :rtype: None
        """
        hf.assert_not_none(
            value=self.aggr_loss_dist[idx], name='aggr_loss_dist', logger=logger
        )


#poisson -genpareto

# define Poisson frequency model
frequency = Frequency(
    dist='poisson',
    par={'mu': 10}
    )

# define Poisson severity model
severity = Severity(
    dist='lognormal',
    par={'shape': 1.5, 'scale': 2}
    )

# define PolicyStructure: default is a empty policy, namely infinty-xs-0 XL with no Quota Share.
policystructure = PolicyStructure(
    layers=Layer(
        deductible=2,
        cover=5,
        n_reinst=2,
        reinst_loading=0.2,
        aggr_deductible=4
    )
)

policystructure = PolicyStructure(
        layers=LayerTower(
            Layer(deductible=0, cover=100, aggr_cover=200),
            Layer(deductible=100, cover=200, aggr_cover=300),
            Layer(deductible=100, cover=200, aggr_cover=300)
            ),
        shares=[0.75, 0.65]
        )

# create LossModel with above defined objects.
# aggr_loss_dist_method approach set to 'mc', i.e. Monte Carlo.
lm_mc = LossModel(
    frequency=frequency,
    severity=severity,
    policystructure=policystructure
    )

lm_mc.costing()

# lm_mc.aggr_loss_dist_calculate(
#     aggr_loss_dist_method='mc',
#     n_sim=1e+05,
#     random_state=1
# )

lm_mc.aggr_loss_dist_calculate(
    aggr_loss_dist_method='recursion',
    sev_discr_method='massdispersal',
    n_sev_discr_nodes=20000,
    sev_discr_step=.01,
    n_aggr_dist_nodes=20000
)

lm_mc.cdf(5)
lm_mc.ppf(0.95)
lm_mc.cdf(np.array([5, 6, 4.2]))
lm_mc.ppf(np.array([0.5, 0.6, 0.94]))

lm_mc.print_aggr_loss_method_specs()
lm_mc.print_costing_specs()
lm_mc.print_policy_layer_specs()()

frequency = Frequency(dist='nbinom', par={'n': 100, 'p': .06})
frequency.par_franchise_adjuster(0.9)
frequency.p
frequency.par_franchise_reverter(0.9)
frequency.p

frequency = Frequency(dist='zmpoisson', par={'mu': 100, 'p0m': 0.15})
frequency.par_franchise_adjuster(0.45)
frequency.mu
frequency.p0m
frequency.par_franchise_reverter(0.45)
frequency.mu
frequency.p0m

severity = Severity(
    par= {
        'shape1': 2,
        'shape2': 4,
        'shape3': 0.5,
        'scale': 1.5
    },
    dist='genbeta'
    )
policystructure = PolicyStructure(
        layers=[
            Layer(aggr_deductible=100, aggr_cover=200),
            Layer(aggr_deductible=75, aggr_cover=175)
            ],
        shares=[0.75, 0.65]
        )

lossmodel_SL = LossModel(
    frequency=frequency,
    severity=severity,
    policystructure=policystructure
    )

# using the aggr_loss_dist_calculate with its arguments.
lossmodel_SL.aggr_loss_dist_calculate(
    aggr_loss_dist_method='mc',
    n_sim=1e+05,
    random_state=1
)
# the resulting aggregate loss distribution is a list with two entries,
# one for each layer of the policystructure.
lossmodel_SL.aggr_loss_dist[0]
lossmodel_SL.aggr_loss_dist[1]