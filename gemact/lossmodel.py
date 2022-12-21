from .libraries import *
from . import config
from . import helperfunctions as hf
from . import distributions as distributions

quick_setup()
logger = log.name('lossmodel')


class PolicyStructure:
    """
    Policy structure component of a loss model.

    :param layers: Non-proportional layer (default is infinity-xs-0 layer).
    :type layers: ``Layer``, ``LayerTower``, ``list``
    :param share: Proportional ceded share (default is 1).
    :type share: ``float``
    """

    def __init__(self, layers=None, shares=None):
        self.layers = layers if layers is not None else Layer()
        self.shares = shares if shares is not None else 1.00
        self._share_and_layer_check()

    @property
    def layers(self):
        return self.__layers

    @layers.setter
    def layers(self, value):
        hf.assert_type_value(
            value, 'layer',
            type=(LayerTower, Layer, list), logger=logger
            )
        if isinstance(value, Layer):
            value = LayerTower(value, is_tower=False)
        elif isinstance(value, list):
            value = LayerTower(*value, is_tower=False)
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
                    lower_bound=0, lower_close=False
                )
        else:
            hf.assert_type_value(
                value, 'share', type=(float, int), logger=logger,
                upper_bound=1, upper_close=True,
                lower_bound=0, lower_close=False
                )
            value = [value]
        self.__shares = value

    @property
    def length(self):
        return len(self.layers)

    def _share_and_layer_check(self):
        """
        Check correctness and consistency between share and layer policy component.
        """
        if len(self.shares) != len(self.layers):
            hf.assert_equality(len(self.shares), 1, 'shares', logger)
            logger.info('Remark: shares value is applied to all layers items')
            self.shares = [self.shares] * len(self.layers)

class Layer:
    """
    Policy structure non-proportional layer of a loss model.

    :param deductible: deductible, also referred to as retention or priority (default value is 0).
    :type deductible: ``int`` or ``float``
    :param cover: cover, also referred to as limit (default value is infinity). Cover plus deductible is the upper priority or severity 'exit point'.
    :type cover: ``int`` or ``float``
    :param aggr_deductible: aggregate deductible, also referred to as aggregate priority or retention (default value is 0).
    :type aggr_deductible: ``int`` or ``float``
    :param aggr_cover: aggregate cover, also referred to as aggregate limit (default is None). Alternative to n_reinst and reinst_loading.
    :type aggr_cover: ``int`` or ``float``
    :param n_reinst: Number of reinstatements (default value is 0).
                    When reinstatements are free (loadings = 0), an alternative parametrization is aggregate cover = (number of reinstatement + 1) * cover.
                    E.g. When the number of reinstatements = 0, the aggregate cover is equal to the cover,
                    when number of reinstatements is infinity there is no aggregate cover
                    (the aggregate cover is infinity).
    :type n_reinst: ``int``
    :param reinst_loading: loadings of reinstatements layers (default value is 0), typically a value in [0, 1].
    :type reinst_loading: ``int`` or ``float`` or ``np.array``
    """

    def __init__(
        self,
        cover=float('inf'),
        deductible=0,
        aggr_deductible=0,
        aggr_cover=None,
        n_reinst=0,
        reinst_loading=0,
        basis=None
        ):

        self.cover = cover
        self.deductible = deductible
        self.aggr_deductible = aggr_deductible
        self.aggr_cover = aggr_cover
        self.n_reinst = n_reinst
        self.reinst_loading = reinst_loading
        self.basis = basis

    @property
    def name(self):
        output = ('%s-xs-%s, %s-xs-%s in aggr.' %(self.cover, self.deductible, self.aggr_cover, self.aggr_deductible))
        return output

    @property
    def deductible(self):
        return self.__deductible

    @deductible.setter
    def deductible(self, value):
        hf.assert_type_value(value, 'deductible', logger=logger, type=(int, float), lower_bound=0)
        self.__deductible = float(value)

    @property
    def cover(self):
        return self.__cover

    @cover.setter
    def cover(self, value):
        hf.assert_type_value(value, 'cover', logger, type=(int, float),
        lower_bound=0, lower_close=False)
        self.__cover = float(value)

    @property
    def aggr_deductible(self):
        # aggregate deductible or priority
        return self.__aggr_deductible

    @aggr_deductible.setter
    def aggr_deductible(self, value):
        hf.assert_type_value(value, 'aggr_deductible', logger, type=(int, float),
        lower_bound=0)
        self.__aggr_deductible = value

    @property
    def n_reinst(self):
        return self.__n_reinst

    @n_reinst.setter
    def n_reinst(self, value):
        if value is not None:
            hf.assert_type_value(value, 'n_reinst', logger, type=(int, float), lower_bound=0)
        self.__n_reinst = value

    @property
    def reinst_loading(self):
        return self.__reinst_loading

    @reinst_loading.setter
    def reinst_loading(self, value):
        name = 'reinst_loading'
        if self.n_reinst == float('inf'):
            logger.warning('Reinstatement loading is disregarded as number of reinstatements is set to infinity.')
            value = np.array([0])  # np.repeat(value, 0)
        else:
            hf.assert_type_value(
                value, name, logger, type=(int, float, np.ndarray),
                lower_bound=0, upper_bound=1
                )
            if isinstance(value, (int, float)):
                value = np.repeat(value, self.n_reinst)
            elif isinstance(value, np.ndarray):
                hf.assert_equality(
                    value.shape[0], self.n_reinst,
                    name, logger
                )
        self.__reinst_loading = value

    @property
    def aggr_cover(self):
        # 'no information about reinst_loading is provided here.'
        return (self.n_reinst + 1) * self.cover
            
    @aggr_cover.setter
    def aggr_cover(self, value):
        if value is not None:
            hf.assert_type_value(
                    value, 'aggr_cover', logger,
                    type=(int, float), lower_bound = self.cover
                    )
            self.n_reinst = (value / self.cover) - 1
            self.reinst_loading = 0
            logger.warning('setting n_reinst to %s and reinst_loading to 0.' % self.n_reinst)
    
    @property
    def basis(self):
        return self.__basis

    @basis.setter
    def basis(self, value):
        if value is not None:
            hf.assert_member(value, config.POLICY_LAYER_BASIS, logger)
        self.__basis = value

    @property
    def exit_point(self):
        return self.deductible + self.cover
    
    @property
    def identifier(self):
        output = '{}_{}_{}_{}_{}'.format(
            self.deductible,
            self.cover,
            self.aggr_deductible,
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
            'aggr_cover', 'n_reinst', 'exit_point', 'basis'
            }

    @staticmethod
    def from_aggregate_cover_to_reinst_converter(aggr_cover, cover):
        """
        Method (static) converting an aggregate cover into its equivalent reinstatement parametrization.

        :param aggr_cover: aggregate cover, also referred to as aggregate limit.
        :type aggr_cover: ``int`` or ``float``
        :param cover: cover, also referred to as limit.
        :type cover: ``int`` or ``float``
        :return: reinstatement parametrization.
        :rtype: ``dict``
        """
        n_reinst = (aggr_cover / cover) - 1
        return {'n_reinst': n_reinst, 'reinst_loading': 0}

class LayerTower(list):
    """
    Policy structure tower of non-proportional layer of a loss model.

    :param maintenance_deductible: maintenance deductible of the layer tower (default is 0).
    :type maintenance_deductible: ``int`` or ``float``
    :param deductible_type: type of deductible (default is 'non-ranking eel'). One of 'eel' (ranking each-and-every-loss) and 'non-ranking eel'(non-ranking each-and-every-loss). 
    :type deductible_type: ``string``

    :param \\**args:
        See below

    :Keyword Arguments:
        * *args* (``Layers``) --
          Layer tower elements.
    """

    def __init__(self, *args, maintenance_deductible=None, deductible_type=None, is_tower=True):
        for arg in args:
            hf.assert_type_value(arg, 'item', logger, Layer)
        # self.remove_duplicates(args)
        super(LayerTower, self).__init__(args)
        # self.extend(args)
        self.manteinance_deductible = 0 if maintenance_deductible is None else maintenance_deductible
        self.deductible_type = 'non-ranking eel' if deductible_type is None else deductible_type
        self.is_tower = is_tower
        self.check_structure()

    @property
    def is_tower(self):
        return self.__is_tower

    @is_tower.setter
    def is_tower(self, value):
        hf.assert_type_value(value, 'is_tower', logger, bool)
        self.__is_tower = value
           
    @property
    def manteinance_deductible(self):
        return self.__manteinance_deductible   

    @manteinance_deductible.setter
    def manteinance_deductible(self, value):
        hf.assert_type_value(value, 'manteinance_deductible', logger=logger, type=(int, float), lower_bound=0)
        self.__manteinance_deductible = value

    @property
    def deductible_type(self):
        return self.__deductible_type

    @deductible_type.setter
    def deductible_type(self, value):
        hf.assert_member(value, config.DEDUCTIBLE_TYPE, logger)
        self.__deductible_type = value

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

    def check_structure(self):
        if self.is_tower:
            self.remove_duplicates()
            self.sort()
            for i in range(1, len(self)):
                name = 'deductible of ' + self[i].name
                hf.assert_equality(
                    self[i].deductible,
                    self[i-1].deductible + self[i-1].cover,
                    name, logger
                )

    def remove_duplicates(self):
        memory = []
        for element in self:
            if element.identifier not in memory:
                memory.append(element.identifier)
            else:
                self.remove(element)
                logger.warning(
                    'Removing %s as a duplicate of another Layer' %(element.name)
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
        hf.assert_member(value, config.DIST_DICT.keys(), logger, config.SITE_LINK)
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
        hf.assert_member(value, config.DIST_DICT.keys(), logger, config.SITE_LINK)
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
    
    def excess_frequency(self, x, base_frequency):
        """
        Excess frequency function, frequency in excess of a given threshold.

        :param x: severity size where excess frequency is evaluated.
        :type x: ``float``
        :param base_frequency: frequency at 0.
        :type base_frequency: ``int``, ``float``
        :return: excess frequency at x.
        :rtype: ``numpy.float`` or ``float``
        """
        hf.assert_type_value(x, 'x', logger, (float, int, np.floating), lower_bound=0, lower_close=False)
        return base_frequency * self.model.sf(x)
 
    def stop_loss_transformation(self, deductible, cover, size=50000, random_state=None):
        """
        Stop loss transformation function.
        General method for severity class, overridden by distirbution specific implementation if available.
        
        :param deductible: deductible, also referred to as retention or priority.
        :type deductible: ``int``, ``float``
        :param cover: cover, also referred to as limit. cover plus deductible is the upper priority or severity 'exit point'.
        :type cover: ``int``, ``float``
        :param size: inner random variates sample size to approximate the integral (default is 50000).
        :type size: ``int``, optional
        :param random_state: inner random state for the random number generator (default=None).
        :type random_state: ``int``, optional
        :return: stop_loss_transformation value.
        :rtype: ``numpy.float``

        """

        output = self.model.stop_loss_transformation(
            deductible,
            cover,
            size,
            random_state
            )
        if isinstance(output, object):
            hf.assert_type_value(deductible, 'deductible', logger, type=(int, float), lower_bound=0)
            hf.assert_type_value(cover, 'cover', logger, type=(int, float), lower_bound=0, lower_close=False)
            hf.assert_type_value(size, 'size', logger, type=(int, float), lower_bound=1, lower_close=False)
            size = int(size)

            random_state = hf.handle_random_state(random_state, logger)
            np.random.seed(random_state)

            t_ = np.random.uniform(low=deductible, high=deductible+cover, size=size)
            output = np.sum(self.model.sf(t_))
            logger.info('Approximating Stop loss transformation function via MC integration')

        return output

    def discretize(
        self,
        discr_method,
        n_discr_nodes,
        discr_step,
        deductible,
        cover
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

        if exit_point != float('inf'):
            n_discr_nodes = n_discr_nodes - 1
            discr_step = (exit_point - deductible) / (n_discr_nodes + 1)
            logger.info('discr_step set to cover/n_sev_discr_nodes.')

        hf.assert_type_value(discr_step, 'discr_step', logger, type=(int, float), lower_bound=0, lower_close=False)
        discr_step = float(discr_step)
        
        if discr_method == 'massdispersal':
            return self._mass_dispersal(deductible, exit_point, discr_step, n_discr_nodes)
        elif discr_method == 'localmoments':
            return self._local_moments(deductible, exit_point, discr_step, n_discr_nodes)

    def _mass_dispersal(self, deductible, exit_point, discr_step, n_discr_nodes):
        """
        Severity discretization according to the mass dispersal method.

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
        f0 = (self.model.cdf(deductible + discr_step / 2) - self.model.cdf(deductible)) / \
             (1 - self.model.cdf(deductible))
        nodes = np.arange(0, n_discr_nodes) + .5

        fj = np.append(
            f0,
            (self.model.cdf(deductible + nodes * discr_step)[1:] -
            self.model.cdf(deductible + nodes * discr_step)[:-1]) /
            (1 - self.model.cdf(deductible))
        )
        if exit_point != float('inf'):
            fj = np.append(fj, (1 - self.model.cdf(
                exit_point - discr_step / 2)) / (
                    1 - self.model.cdf(deductible)))

        nodes = self.loc + np.arange(0, n_discr_nodes) * discr_step

        if exit_point != float('inf'):
            nodes = np.concatenate((nodes, [nodes[-1] + discr_step]))

        return {'nodes': nodes, 'fj': fj}

    def _upper_discr_point_prob_adjuster(self, deductible, exit_point, discr_step):
        """
        Probability of the discretization upper point in the local moment.
        In case an upper priority on the severity is provided, the probability of the node sequence upper point
        is adjusted to be coherent with discretization step size and number of nodes.

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
            output = (self.model.lev(exit_point - self.loc) -
                self.model.lev(exit_point - self.loc - discr_step)) / \
                    (discr_step * self.model.den(low=deductible, loc=self.loc))
        return output

    def _local_moments(self, deductible, exit_point, discr_step, n_discr_nodes):
        """
        Severity discretization according to the local moments method.

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

        last_node_prob = self._upper_discr_point_prob_adjuster(deductible, exit_point, discr_step)

        n = self.model.lev(deductible + discr_step - self.loc) - self.model.lev(
            deductible - self.loc)
        den = discr_step * self.model.den(low=deductible, loc=self.loc)
        nj = 2 * self.model.lev(deductible - self.loc + np.arange(
            1, n_discr_nodes) * discr_step) - self.model.lev(
            deductible - self.loc + np.arange(
                0, n_discr_nodes - 1) * discr_step) - self.model.lev(
            deductible - self.loc + np.arange(2, n_discr_nodes + 1) * discr_step)

        fj = np.append(1 - n / den, nj / den)

        nodes = self.loc + np.arange(0, n_discr_nodes) * discr_step
        if exit_point != float('inf'):
            nodes = np.concatenate((nodes, [nodes[-1] + discr_step]))
        return {'nodes': nodes, 'fj': np.append(fj, last_node_prob)}

class LossModel:
    """
    Aggregate loss model for (re)insurance pricing and risk modeling using a collective risk model framework.

    :param aggr_loss_dist_method: computational method to approximate the aggregate loss distribution.
                                  One of Fast Fourier Transform ('fft'),
                                  Panjer recursion ('recursion') and Monte Carlo simulation ('mc').
    :type aggr_loss_dist_method: ``str``
    :param n_sim: number of simulations of Monte Carlo (mc) method for the aggregate loss distribution approximation.
    :type n_sim: ``int``
    :param tilt: whether tilting of FFT is present or not (default is 0).
    :type tilt: ``bool``
    :param tilt_value: tilting parameter value of FFT method for the aggregate loss distribution approximation.
    :type tilt_value: ``float``
    :param random_state: random state for the random number generator in MC.
    :type random_state: ``int``, optional
    :param n_aggr_dist_nodes: number of nodes in the approximated aggregate loss distribution.
    :type n_aggr_dist_nodes: ``int``
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
        self.aggr_loss_dist_calculate()

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

    def aggr_loss_dist_calculate(
        self,
        aggr_loss_dist_method=None,
        n_aggr_dist_nodes=None,
        n_sim=None,
        random_state=None,
        tilt=None,
        tilt_value=None,
        sev_discr_method=None
        ):
        """
        Approximation of aggregate loss distributions by calculating nodes, pdf, and cdf.
        Distributions can be accessed via the aggr_loss_dist property,
        which is a list of ``dict``, each one representing a aggregate loss distribution.

        :param aggr_loss_dist_method: computational method to approximate the aggregate loss distribution.
                                      One of Fast Fourier Transform ('FFT'), Panjer recursion ('Recursion')
                                      and Monte Carlo simulation ('mc'), optional (default 'mc').
        :type aggr_loss_dist_method: ``str``
        :param n_aggr_dist_nodes: number of nodes in the approximated aggregate loss distribution.
        :type n_aggr_dist_nodes: ``int``
        :param n_sim: number of simulations of Monte Carlo (mc) method
                      for the aggregate loss distribution approximation, optional (default is 10000).
        :type n_sim: ``int``
        :param random_state: random state for the random number generator in MC, optional.
        :type random_state: ``int``
        :param sev_discr_method: severity discretization method, optional (default is 'localmoments').
        :type sev_discr_method: ``str``
        :param tilt: whether tilting of FFT is present or not, optional (default is 0).
        :type tilt: ``bool``
        :param tilt_value: tilting parameter value of FFT method for the aggregate loss distribution approximation,
                           optional.
        :type tilt_value: ``float``
        :return: Void
        :rtype: ``None``
        """

        if (aggr_loss_dist_method is None) and (self.aggr_loss_dist_method is None):
            aggr_dist_list = [{'nodes': None, 'epdf': None, 'ecdf': None}]
        else:
            aggr_dist_list = []
            for i in range(self.policystructure.length):
                # backup original frequency model (to later restore) and
                # adjust the frequency model for the calculation
                frequency_model_bak = self.frequency.model
                self.frequency.model.par_franchise_adjuster(
                    1 - self.severity.model.cdf(
                        self.policystructure.layers[i].deductible
                        )
                    )

                if aggr_loss_dist_method is not None:
                    self.aggr_loss_dist_method = aggr_loss_dist_method

                if n_aggr_dist_nodes is not None:
                    self.n_aggr_dist_nodes = n_aggr_dist_nodes

                if self.aggr_loss_dist_method == 'mc':
                    if n_sim is not None:
                        self.n_sim = n_sim
                    if random_state is not None:
                        self.random_state = random_state
                    aggr_dist_list.append(
                        self._mc_simulation(
                            self.policystructure.layers[i].deductible,
                            self.policystructure.layers[i].exit_point
                            )
                    )
                else:
                    sevdict = self.severity.discretize(
                        self.sev_discr_method,
                        self.n_sev_discr_nodes,
                        self.sev_discr_step,
                        self.policystructure.layers[i].deductible,
                        self.policystructure.layers[i].cover
                        )
                    
                    # adjustment of the discr_step to use in the aggregate distribution calculation below.
                    # severity.discretize() method performs it within its body.
                    if self.policystructure.layers[i].exit_point != float('inf'):
                        discr_step = (self.policystructure.layers[i].cover) / self.n_sev_discr_nodes
                        logger.info('discr_step set to cover/n_sev_discr_nodes.') 

                    if self.aggr_loss_dist_method == 'recursion':
                        if sev_discr_method is not None:
                            self.sev_discr_method = sev_discr_method
                        aggr_dist_list.append(
                            self._panjer_recursion(
                                sevdict, discr_step
                                )
                        )
                    else:  # self.aggr_loss_dist_method == 'fft'
                        if tilt is not None:
                            self.tilt = tilt
                        if tilt_value is not None:
                            self.tilt_value = tilt_value
                        if sev_discr_method is not None:
                            self.sev_discr_method = sev_discr_method
                        aggr_dist_list.append(
                            self._fft(
                                sevdict, discr_step
                                )
                        )
                # restore original unadjusted frequency model
                self.frequency.model = frequency_model_bak
        self.__aggr_loss_dist = aggr_dist_list
        return

    def _fft(self, sevdict, discr_step):
        """
        Aggregate loss distribution via fast Fourier transform.

        :return: aggregate loss distribution empirical pdf, cdf, nodes
        :rtype: ``dict``
        """
        logger.info('..Approximating aggregate loss distribution via FFT..')

        fj = sevdict['fj']

        if self.tilt:
            tilting_par = 20 / self.n_aggr_dist_nodes if self.tilt_value is None else self.tilt_value
        else:
            tilting_par = 0

        # if exit_point != float('inf'):
        #     n_discr_nodes = self.n_sev_discr_nodes - 1
        #     discr_step = (exit_point - deductible) / (n_discr_nodes + 1)
        #     logger.info('discr_step set to cover/n_sev_discr_nodes.')

        # if exit_point == float('inf'):
        fj = np.append(fj, np.repeat(0, self.n_aggr_dist_nodes - self.n_sev_discr_nodes))
        # else:
        # fj = np.append(fj, np.repeat(0, self.n_aggr_dist_nodes - self.n_sev_discr_nodes - 1))

        f_hat = fft(np.exp(-tilting_par * np.arange(0, self.n_aggr_dist_nodes, step=1)) * fj)
        g_hat = self.frequency.model.pgf(f=f_hat)
        g = np.exp(tilting_par * np.arange(0, self.n_aggr_dist_nodes, step=1)) * np.real(ifft(g_hat))

        logger.info('..Distribution via FFT completed..')

        return {'epdf': g,
                'ecdf': np.cumsum(g),
                'nodes': discr_step * np.arange(0, self.n_aggr_dist_nodes, step=1)}

    def _panjer_recursion(self, sevdict, discr_step):
        """
        Aggregate loss distribution via Panjer recursion.

        :return: aggregate loss distribution empirical pdf, cdf, nodes
        :rtype: ``dict``
        """
        logger.info('..Approximating aggregate loss distribution via Panjer recursion..')

        fj = sevdict['fj']
        a, b, p0, g = self.frequency.abp0g0(fj)

        # if exit_point == float('inf'):
        fj = np.append(fj, np.repeat(0, self.n_aggr_dist_nodes - self.n_sev_discr_nodes))
        # else:
        #    fj = np.append(fj, np.repeat(0, self.n_aggr_dist_nodes - self.n_sev_discr_nodes - 1))

        for j in range(1, self.n_aggr_dist_nodes):
            z = np.arange(1, min(j, len(fj) - 1) + 1)
            g.append((1 / (1 - a * fj[0])) * ((self.frequency.model.pmf(1) - (a + b) * p0) * fj[z[-1]] + np.sum(
                ((a + b * z / j) * fj[z] * np.flip(g)[:len(z)]))))

        logger.info('..Panjer recursion completed..')

        return {'epdf': g,
                'ecdf': np.cumsum(g),
                'nodes': discr_step * np.arange(0, self.n_aggr_dist_nodes, step=1)}

    def _mc_simulation(self, deductible, exit_point):
        """
        Aggregate loss distribution via Monte Carlo simulation.

        :return: aggregate loss distribution empirical pdf, cdf, nodes.
        :rtype: ``dict``
        """
        logger.info('..Approximating aggregate loss distribution via Monte Carlo simulation..')
        
        p0 = self.severity.model.cdf(deductible) if deductible > 1e-05 else 0.

        fqsample = self.frequency.model.rvs(self.n_sim, random_state=self.random_state)
        svsample = self.severity.model.rvs(int(np.sum(fqsample) + np.ceil(p0 * self.n_sim)), random_state=self.random_state)

        if deductible > 1e-05:
            svsample = svsample[svsample > deductible]
            j = 1
            while svsample.shape[0] < self.n_sim:
                n0 = int(np.ceil(p0 * (self.n_sim - svsample.shape[0])))
                svsample = np.concatenate((svsample, self.severity.model.rvs(n0, random_state=self.random_state + j)))
                svsample = svsample[svsample > deductible]
                j += 1
            svsample = svsample - deductible

        if exit_point < float('inf'):
            svsample = np.minimum(svsample, exit_point)

        cs = np.cumsum(fqsample).astype(int)

        if fqsample[0:1] == 0:
            xsim = np.array([0])
        else:
            xsim = np.array([np.sum(svsample[0:cs[0]])])

        for i in range(self.n_sim - 1):
            if cs[i] == cs[i + 1]:
                xsim = np.concatenate((xsim, [0]))
            else:
                xsim = np.concatenate((xsim, [np.sum(svsample[cs[i]:cs[i + 1]])]))

        logger.info('..Simulation completed..')

        x_, ecdf = hf.ecdf(xsim)
        epdf = np.repeat(1 / self.n_sim, self.n_sim)

        return {'epdf': epdf,
                'ecdf': ecdf,
                'nodes': x_}

    def aggr_loss_moment(self, central=False, order=1, idx=0):
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
        self._aggr_loss_dist_check(idx)
        aggr_loss_dist_ = self.aggr_loss_dist[idx]

        hf.assert_type_value(central, 'central', logger, bool)
        hf.assert_type_value(order, 'order', logger, (int, float), lower_bound=0, lower_close=False)
        order = int(order)
        
        if self.aggr_loss_dist_method != 'mc':
            lmmean = np.sum(aggr_loss_dist_['epdf'] * aggr_loss_dist_['nodes'])
            return np.sum(aggr_loss_dist_['epdf'] * ((aggr_loss_dist_['nodes'] - (central * lmmean)) ** order))
        else:
            return np.mean((aggr_loss_dist_['nodes'] - central * np.mean(aggr_loss_dist_['nodes'])) ** order)

    def aggr_loss_ppf(self, q, idx=0):
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
            idx, 'idx', logger, int,
            upper_bound=len(self.aggr_loss_dist)-1,
            lower_bound=0
            )
        self._aggr_loss_dist_check(idx)
        aggr_loss_dist_ = self.aggr_loss_dist[idx]

        for item in q:
            hf.assert_type_value(item, 'q', logger, (float, int), upper_bound=1, lower_bound=0)
        q = np.ravel(q)
        return np.quantile(aggr_loss_dist_['nodes'], q=q)

    def aggr_loss_cdf(self, x, idx=0):
        """
        Aggregate loss distribution cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
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
        self._aggr_loss_dist_check(idx)
        aggr_loss_dist_ = self.aggr_loss_dist[idx]

        x = np.ravel(x)
        x = np.maximum(x, 0)
        y_ = np.append(0, aggr_loss_dist_['ecdf'])
        x_ = np.append(0, aggr_loss_dist_['nodes'])
        output = np.empty(len(x))

        for k in np.arange(len(x)):
            x_temp = x[k]
            j = np.searchsorted(x_, x_temp)
            run = x_[j] - x_[max(j - 1, 0)] if (x_[j] - x_[max(j - 1, 0)]) > 0 else 1
            rise = y_[j] - y_[max(j - 1, 0)]
            output[k] = y_[max(j - 1, 0)] + (x_temp - x_[max(j - 1, 0)]) * rise / run

        return output

    def aggr_loss_rvs(self, size=1, random_state=None, idx=0):
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
        self._aggr_loss_dist_check(idx)
        aggr_loss_dist_ = self.aggr_loss_dist[idx]

        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=0, lower_close=False)
        size = int(size)

        output = np.random.choice(aggr_loss_dist_['nodes'], size=size, p=aggr_loss_dist_['epdf'])
        return output

    def aggr_loss_mean(self, idx=0):
        """
        Mean of the aggregate loss.

        :param idx: list index corresponding to the loss distribution of interest (default is 0).
        :type idx: ``int``
        :return: mean of the aggregate loss.
        :rtype: ``numpy.float64``
        """

        return self.aggr_loss_moment(central=False, order=1, idx=idx)

    def aggr_loss_std(self, idx=0):
        """
        Standard deviation of the aggregate loss.

        :param n: list index corresponding to the loss distribution of interest (default is 0).
        :type n: ``idx``
        :return: standard deviation of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        return self.aggr_loss_moment(central=True, order=2, idx=idx) ** 1 / 2

    def aggr_loss_skewness(self, idx=0):
        """
        Skewness of the aggregate loss.

        :param idx: list index corresponding to the loss distribution of interest (default is 0).
        :type idx: ``int``
        :return: skewness of the aggregate loss.
        :rtype: ``numpy.float64``
        """
        return self.aggr_loss_moment(central=True, order=3, idx=idx) / self.aggr_loss_moment(central=True, order=2, idx=idx) ** 3 / 2

    def _stop_loss_pricing(self, t, aggr_loss_dist):
        """
        Expected value of a stop loss contract with deductible t, also referred to as retention or priority.

        :param t: deductible.
        :type t: ``float``
        :param aggr_loss_dist: aggrgegate loss distribution of interest.
        :type aggr_loss_dist: ``dict``
        :return: stop loss contract expected value.
        """

        x = aggr_loss_dist['nodes']
        t = np.repeat([t], 1)
        
        if self.aggr_loss_dist_method == 'mc':
            if t.shape[0] == 1:
                return np.mean(np.maximum(x - t, 0))
            else:
                lg = t.shape[0]
                obs = x.shape[0]
                v1_ = np.tile(x, lg).reshape(lg, -1) - np.repeat(t, obs).reshape(lg, -1)
                v1_[v1_ < 1e-6] = .0
                return np.apply_along_axis(arr=v1_, func1d=np.mean, axis=1)
        else:
            probs = aggr_loss_dist['epdf']
            if t.shape[0] == 1:
                return np.sum(np.maximum(x - t, 0) * probs)
            else:
                lg = t.shape[0]
                obs = x.shape[0]
                v1_ = np.tile(x, lg).reshape(lg, -1) - np.repeat(t, obs).reshape(lg, -1)
                v1_[v1_ < 1e-6] = .0
                v2_ = v1_ * np.tile(probs, lg).reshape(lg, -1)
                return np.apply_along_axis(arr=v2_, func1d=np.sum, axis=1)

    def _reinstatement_pricing(
        self,
        aggr_loss_dist,
        aggr_deductible,
        n_reinst,
        cover,
        reinst_loading
        ):
        """
        Reinstatements pricing.

        :return: reinstatements pricing.
        :rtype: ``numpy.ndarray``
        """

        output = self._stop_loss_pricing(
            aggr_deductible,
            aggr_loss_dist
            ) - self._stop_loss_pricing(
            aggr_deductible + (n_reinst + 1) * cover,
            aggr_loss_dist
            )
        if n_reinst > 0:
            lower_k = np.linspace(start=0, stop=n_reinst, num=n_reinst + 1)
            dlk = (
                self._stop_loss_pricing(
                aggr_deductible + lower_k[:-1] * cover,
                aggr_loss_dist
                ) - self._stop_loss_pricing(
                aggr_deductible + lower_k[1:] * cover,
                aggr_loss_dist
                )
                )
            den = 1 + np.sum(dlk * reinst_loading) / cover
            output = output / den
        return output

    def pricing(self, idx=0):
        """
        Actuarial pricing (also referred to as costing) for proportional and non-proportional reinsurance covers,
        such as quota share, excess-of-loss (including and excluding reinstatements) and stop loss.

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
        self._aggr_loss_dist_check(idx)
        aggr_loss_dist_ = self.aggr_loss_dist[idx]
        n_reinst_ = self.policystructure.layers[idx].n_reinst
        aggr_deductible_ = self.policystructure.layers[idx].aggr_deductible
        deductible_ = self.policystructure.layers[idx].deductible
        cover_ = self.policystructure.layers[idx].cover
        reinst_loading_ = self.policystructure.layers[idx].reinst_loading
        exit_point_ = self.policystructure.layers[idx].exit_point
        alpha_qs_ = self.policystructure.shares[idx]

        p_ = None
        if n_reinst_ == float('inf'):
            p_ = self._stop_loss_pricing(
                t=aggr_deductible_,
                aggr_loss_dist=aggr_loss_dist_
                )
        else: # n_reinst_ != float('inf')
            p_ = self._reinstatement_pricing(
                aggr_loss_dist_,
                aggr_deductible_,
                n_reinst_,
                cover_,
                reinst_loading_
                )

        data = [
            ['Deductible', 'd', deductible_],
            ['Cover', 'c', cover_],
            ['Upper priority', 'c+d', exit_point_],
            ['Aggregate deductible', 'v', aggr_deductible_],
            ['Quota share portion', 'a', alpha_qs_]
        ]
        if n_reinst_ != float('inf'):
            data.extend([['Reinstatements (no.)', 'K', n_reinst_]])

        data.append(['Pure premium', 'P', alpha_qs_ * p_])
        print('{: >20} {: >20} {: >20} {: >20}'.format(' ', *['Contract specification', 'parameter', 'value']))
        print('{: >20} {: >20}'.format(' ', *['====================================================================']))
        for row in data:
            print('{: >20} {: >20} {: >20} {: >20}'.format('', *row))
        print('\n Reinstatement layer loading l: ', reinst_loading_)
        if self.aggr_loss_dist_method == 'mc':
            print(self.aggr_loss_dist_method, '\t n_sim: ', self.n_sim, '\t random_state:', self.random_state)
        else:
            n_sev2print = self.n_sev_discr_nodes
            # if cover_ != float('inf'):
            #     n_sev2print += 1
            print(
                self.aggr_loss_dist_method, '\t n_sev_discr_nodes m: ',
                n_sev2print, '\t n_aggr_dist_nodes n: ',
                self.n_aggr_dist_nodes)

    def print_aggr_loss_specs(self, idx=0):
        """
        Print aggregate loss distribution approximation specifications.

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
        cover_ = self.policystructure.layers[idx].cover

        data = [
            ['aggr_loss_dist_method', self.aggr_loss_dist_method],
            ['n_aggr_dist_nodes', self.n_aggr_dist_nodes],
        ]

        if self.aggr_loss_dist_method == 'mc':
            data.extend([
                ['n_sim', self.n_sim],
                ['random_state', self.random_state],
            ])
        elif self.aggr_loss_dist_method == 'fft':
            data.extend([
                ['tilt', self.tilt],
                ['tilt_value', self.tilt_value],
            ])

        n_sev2print = self.n_sev_discr_nodes

        if cover_ != float('inf'):
            n_sev2print += 1

        data.extend([
            ['sev_discr_method', self.sev_discr_method],
            ['sev_discr_step', self.sev_discr_step],
            ['n_sev_discr_nodes', n_sev2print]
        ])

        print('{: >20} {: >20} {: >20}'.format(' ', *['         ', 'value']))
        print('{: >20} {: >20}'.format(' ', *[' ==================================================================']))
        for row in data:
            print('{: >20} {: >20} {: >20}'.format('', *row))
        return

    def print_policy_specs(self, idx=0):
        """
        Print policy structure specifications.

        :param idx: index corresponding to the policystructure element of interest (default is 0).
        :type idx: ``int``
        :return: Void
        :rtype: None
        """

        n_reinst_ = self.policystructure.layers[idx].n_reinst
        aggr_deductible_ = self.policystructure.layers[idx].aggr_deductible
        deductible_ = self.policystructure.layers[idx].deductible
        cover_ = self.policystructure.layers[idx].cover
        reinst_loading_ = self.policystructure.layers[idx].reinst_loading
        alpha_qs_ = self.policystructure.shares[idx]

        data = [
            ['Deductible', deductible_],
            ['Cover', cover_],
            ['Aggregate deductible', aggr_deductible_],
            ['Quota share portion', alpha_qs_],
            ['Reinstatements (no.)', n_reinst_]
        ]


        print('{: >20} {: >20} {: >20}'.format(' ', *['Contract specification', 'value']))
        print('{: >20} {: >20}'.format(' ', *[' ==================================================================']))
        for row in data:
            print('{: >20} {: >20} {: >20}'.format(' ', *row))
        print('\n Reinstatement layer loading c: ', reinst_loading_)
        return

    def _aggr_loss_dist_check(self, idx=0):
        """
        Assert whether the aggregate loss distribution is not missing.
        Helper method called before executing other methods based on ``aggr_loss_dist`` property.

        :param idx: index corresponding to the policystructure element of interest (default is 0).
        :type idx: ``int``
        :return: Void
        :rtype: None
        """
        if isinstance(self.aggr_loss_dist[idx], type(None)):
            logger.error('Make sure to use aggr_loss_dist_calculate method first')
            raise TypeError
