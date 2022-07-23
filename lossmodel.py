import numpy as np
from scipy.fft import fft, ifft
from . import helperfunctions as hf
from . import config
from twiggy import quick_setup, log
import time  # used for setting random number generator seed if None

quick_setup()
logger = log.name('lossmodel')


# Loss model Frequency component
class _Frequency:
    """
    Class representing the frequency component of the loss models underlying the collective risk model.

    :param \** kwargs:
    See below

    :Keyword Arguments:
        * *fq_dist* (``str``) --
          Name of the frequency distribution (default is 'poisson' model).
        * *fq_par* (``dict``) --
          parameters of the frequency distribution (default is 'poisson' model).

    """

    def __init__(self, **kwargs):
        self.fq_dist = kwargs.get('fq_dist', 'poisson')
        self.fq_par = kwargs.get('fq_par', {'mu': 1})
        self.fq_model = self.fq_dist(**self.fq_par)

    @property
    def fq_dist(self):
        return self.__fq_dist

    @fq_dist.setter
    def fq_dist(self, value):
        assert value in config.DIST_DICT.keys(), '%r is not supported.\n See %s' % (value, config.SITE_LINK)
        self.__fq_dist = config.DIST_DICT[value]

    @property
    def fq_par(self):
        return self.__fq_par

    @fq_par.setter
    def fq_par(self, value):
        assert isinstance(value, dict), logger.info('Distribution parameters must be given as a dictionary')

        try:
            self.fq_dist(**value)
        except:
            logger.error('Distribution not correctly parametrized.\n See %s' % config.SITE_LINK)

        if 'zm' in self.fq_dist(**value).category() and 'loc' not in value.keys():
            value['loc'] = -1

        self.__fq_par = value

    @property
    def p0(self):
        output = self.fq_par['p0M'] if 'zm' in self.fq_model.category() else None
        return output

    @property
    def fq_model(self):
        return self.__fq_model

    @fq_model.setter
    def fq_model(self, value):

        try:
            assert 'frequency' in value.category(), logger.error(
                '%r is not a valid frequency model' % value)
        except:
            logger.error('Please provide a correct frequency model.\n See %s' % config.SITE_LINK)

        self.__fq_model = value

    def abp0g0(self, fj):
        """
        Parameters of the frequency distribution according to the (a, b, k) parametrization,
        the probability generating function computed in zero given the discrete severity probs,
        and the probability of the distribution in zero.

        :param fj: discretized severity distribution probabilities.
        :type fj: numpy.ndarray
        """
        a, b, p0 = self.fq_model.abk()
        return a, b, p0, [self.fq_model.pgf(fj[0])]


# Loss model Severity component
class _Severity:
    """
    Class representing the severity component of the loss models underlying the collective risk model.
    
    :param sev_discr_method: severity discretization method (default is 'localmoments').
    :type sev_discr_method: ``str``, optional
    :param sev_discr_step: severity discretization step.
    :type sev_discr_step: ``float``
    :param n_sev_discr_nodes: number of nodes of the discretized severity.
    :type n_sev_discr_nodes: ``int``
    :param deductible: deductible, also referred to as retention or priority (default value is 0).
    :type deductible: ``int`` or ``float``
    :param cover: contract cover, also referred to as limit,
                  deductible plus cover is the contract upper priority or severity 'exit point'
                  (default value is infinity).
    :type cover: ``int`` or ``float``
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *sev_dist* (``str``) --
          Name of the severity distribution.
          See the distribution module for the continuous distributions supported in GEMAct.
        * *sev_par* (``dict``) --
          parameters of the severity distribution.
    """

    def __init__(
            self,
            sev_discr_method='localmoments',
            sev_discr_step=None,
            n_sev_discr_nodes=None,
            deductible=0,
            cover=float('inf'),
            **kwargs
    ):

        self.sev_dist = kwargs.get('sev_dist', 'lognormal')
        self.sev_par = kwargs.get('sev_par', {'s': 1})
        self.sev_model = self.sev_dist(**self.sev_par)
        self.sev_discr_method = sev_discr_method
        self.deductible = deductible
        self.cover = cover

        self.n_sev_discr_nodes = n_sev_discr_nodes
        self.sev_discr_step = sev_discr_step
        self.loc = self.sev_par['loc'] if 'loc' in self.sev_par.keys() else .0

    @property
    def sev_discr_method(self):
        return self.__sev_discr_method

    @sev_discr_method.setter
    def sev_discr_method(self, value):
        assert value in config.SEV_DISCRETIZATION_METHOD_LIST, '%r is not one of %s' % (
        value, config.SEV_DISCRETIZATION_METHOD_LIST)
        self.__sev_discr_method = value

    @property
    def sev_dist(self):
        return self.__sev_dist

    @sev_dist.setter
    def sev_dist(self, value):
        assert value in config.DIST_DICT.keys(), '%r is not supported.\n See %s' % (value, config.SITE_LINK)
        self.__sev_dist = config.DIST_DICT[value]

    @property
    def sev_par(self):
        return self.__sev_par

    @sev_par.setter
    def sev_par(self, value):

        assert isinstance(value, dict), 'Distribution parameters must be given as a dictionary'

        try:
            self.sev_dist(**value)
        except:
            logger.error('Distribution not correctly parametrized.\n See %s' % config.SITE_LINK)

        self.__sev_par = value

    @property
    def deductible(self):
        return self.__deductible

    @deductible.setter
    def deductible(self, value):
        assert isinstance(value, (int, float)), logger.error('%r is not an int or a float' % value)
        assert value >= 0, logger.error('Deductible must be larger than or equal to zero')
        self.__deductible = float(value)

    @property
    def cover(self):
        return self.__cover

    @cover.setter
    def cover(self, value):
        assert isinstance(value, (int, float)), logger.error('%r is not an int or a float' % value)
        assert value >= 0, logger.error('Cover must be larger than or equal to 0')
        self.__cover = float(value)

    @property
    def n_sev_discr_nodes(self):
        return self.__n_sev_discr_nodes

    @n_sev_discr_nodes.setter
    def n_sev_discr_nodes(self, value):

        if value is not None:
            try:
                if not type(value) == type(int(value)):
                    assert value > 0, logger.error('Number of discretization steps must be a postive integer.')
                    logger.warning('Value set to integer.')
                    value = int(value)
                else:
                    value = value

                if self.u != float('inf'):
                    value = value - 1

            except:
                logger.error('Number of discretization steps must be a postive integer.')

        self.__n_sev_discr_nodes = value

    @property
    def sev_discr_step(self):
        return self.__sev_discr_step

    @sev_discr_step.setter
    def sev_discr_step(self, value):
        if value is None:
            assert (self.m is None), logger.error('Missing discretization step.')
        else:
            if self.u != float('inf'):
                logger.info('Discretization step set to (u-d)/m.')
                value = (self.u - self.d) / (self.m + 1)
            else:
                assert value > 0, logger.error('Discretization step must be larger than zero.')
                value = float(value)
        self.__sev_discr_step = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        self.__loc = value

    @property
    def sev_model(self):
        return self.__sev_model

    @sev_model.setter
    def sev_model(self, value):
        try:
            assert 'severity' in value.category(), logger.error(
                '%r is not a valid severity model' % value
            )
        except:
            logger.error('Please provide a correct severity model.\n See %s' % config.SITE_LINK)

        self.__sev_model = value

    @property
    def h(self):
        return self.sev_discr_step

    @property
    def d(self):
        return self.deductible

    @property
    def m(self):
        return self.n_sev_discr_nodes

    @property
    def u(self):
        return self.deductible + self.cover

    # def _cover_check(self):
    #     assert isinstance(self.cover, (int, float)), logger.error('%r is not an int or a float' %self.cover)
    #     assert self.cover >= 0, logger.error('Cover must be larger than or equal to 0')

    # def _stop_loss_transformation(self):
    #     print('Work in progress. Soon available')
    #     pass

    def severity_discretization(self):
        if self.sev_discr_method not in config.SEV_DISCRETIZATION_METHOD_LIST:
            raise ValueError('%r is not one of %s.' % (self.sev_discr_method, config.SEV_DISCRETIZATION_METHOD_LIST))

        if self.sev_discr_method == 'massdispersal':
            return self._mass_dispersal()
        elif self.sev_discr_method == 'localmoments':
            return self._local_moments()

    def _mass_dispersal(self):
        """
        Severity discretization according to the mass dispersal method.

        :return: discrete severity
        :rtype: ``dict``
        """
        f0 = (self.sev_model.cdf(self.d + self.h / 2) - self.sev_model.cdf(self.d)) / \
             (1 - self.sev_model.cdf(self.d))
        nodes = np.arange(0, self.m) + .5
        fj = np.append(
            f0,
            (self.sev_model.cdf(self.d + nodes * self.h)[1:] - self.sev_model.cdf(self.d + nodes * self.h)[:-1]) / \
            (1 - self.sev_model.cdf(self.d))
        )
        if self.u != float('inf'):
            fj = np.append(fj, (1 - self.sev_model.cdf(self.u - self.h / 2)) / (1 - self.sev_model.cdf(self.d)))

        nodes = self.loc + np.arange(0, self.m) * self.h

        if self.u != float('inf'):
            nodes = np.concatenate((nodes, [nodes[-1] + self.h]))

        return {'sev_nodes': nodes, 'fj': fj}

    def _upper_discr_point_prob_adjuster(self):
        """
        It calculates the probability of the discretization upper point in the local moment.
        In case an upper priority on the severity is provided, the probability of the node sequence upper point
        is adjusted to be coherent with discretization step size and number of nodes.

        :return: probability mass in (u-d/h)*m
        :rtype: ``numpy.ndarray``
        """

        if self.u == float('inf'):
            output = np.array([])
        else:
            output = (self.sev_model.lev(self.u - self.loc) - self.sev_model.lev(self.u - self.loc - self.h)) / \
                     (self.h * self.sev_model.den(low=self.d, loc=self.loc))
        return output

    def _local_moments(self):
        """
        Severity discretization according to the local moments method.

        :return: discrete severity.
        :rtype: ``dict``

        """

        last_node_prob = self._upper_discr_point_prob_adjuster()
        n = self.sev_model.lev(self.d + self.h - self.loc) - self.sev_model.lev(self.d - self.loc)
        den = self.h * self.sev_model.den(low=self.d, loc=self.loc)
        nj = 2 * self.sev_model.lev(self.d - self.loc + np.arange(1, self.m) * self.h) - self.sev_model.lev(
            self.d - self.loc + np.arange(0, self.m - 1) * self.h) - self.sev_model.lev(
            self.d - self.loc + np.arange(2, self.m + 1) * self.h)

        fj = np.append(1 - n / den, nj / den)

        nodes = self.loc + np.arange(0, self.m) * self.h
        if self.u != float('inf'):
            nodes = np.concatenate((nodes, [nodes[-1] + self.h]))
        return {'sev_nodes': nodes, 'fj': np.append(fj, last_node_prob)}


## Loss Model component
class LossModel(_Severity, _Frequency):
    """
    Class representing the loss model, i.e. a combination of frequency and severity model,
    for (re)insurance pricing and risk modeling using a collective risk model framework.

    :param aggr_loss_dist_method: computational method to approximate the aggregate loss distribution. One of Fast Fourier Transform ('fft'), Panjer recursion ('recursion') and Monte Carlo simulation ('mc').
    :type aggr_loss_dist_method: ``str``
    :param aggr_n_deductible: number of deductible (default = 1) in the stop loss (aggregate) deductible, also referred to as priority or retention. Namely, ``aggr_n_deductible * deductible`` =  aggregate priority.
    :type aggr_n_deductible: ``int``
    :param n_reinst: number of reinstatements layers. Alternative parametrization to aggregate limit cover, i.e. aggregate cover = number of reinstatement * cover
    :type n_reinst: ``int``
    :param reinst_loading: reinstatements layers loading.
    :type reinst_loading: ``float``
    :param alpha_qs: quota share ceded portion.
    :type alpha_qs: ``float``
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
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *fq_dist* (``str``) --
          Frequency model distribution name.
        * *fq_par* (``dict``) --
          Frequency model distribution parameters.
        * *sev_dist* (``str``) --
          Severity model distribution name.
        * *sev_par* (``dict``) --
          Severity model distribution parameters.
        * *sev_discr_method* (``float``) --
          Severity discretization method (default value is 'localmoments').
        * *sev_discr_step* (``float``) --
          Severity model discretization step.
        * *n_sev_discr_nodes* (``int``) --
          Number of nodes of the discretized severity.
        * *deductible* (``int`` or ``float``) --
          Contract deductible, also referred to as retention or priority (default value is 0).
        * *cover* (``int`` or ``float``) --
          Contract cover (deductible plus cover is the contract upper priority or severity 'exit point').
    """

    def __init__(
            self,
            aggr_loss_dist_method=None,
            n_sim=10000,
            tilt=False,
            tilt_value=0,
            random_state=None,
            n_aggr_dist_nodes=20000,
            aggr_n_deductible=1,
            n_reinst=float('inf'),
            reinst_loading=0,
            alpha_qs=1,
            **kwargs
    ):

        # Loss Model Frequency and Severity
        _Frequency.__init__(self, **kwargs)
        _Severity.__init__(self, **kwargs)

        # Loss Model contract features
        self.fq_model.par_franchise_adjuster(self.nu)
        self.aggr_n_deductible = aggr_n_deductible
        self.n_reinst = n_reinst
        self.reinst_loading = reinst_loading
        self.alpha_qs = alpha_qs

        # Loss Model aggregate loss distribution approximation method specifications
        self.aggr_loss_dist_method = aggr_loss_dist_method
        self.n_sim = n_sim
        self.random_state = random_state
        self.n_aggr_dist_nodes = n_aggr_dist_nodes
        self.tilt = tilt
        self.tilt_value = tilt_value
        self.aggr_loss_dist_calculate()

    @property
    def aggr_loss_dist(self):
        return self.__aggr_loss_dist

    @aggr_loss_dist.setter
    def aggr_loss_dist(self, value):
        assert isinstance(value, dict), 'provided aggregate loss distribution should be a dictionary'
        assert set(value.keys()) == {'nodes', 'epdf', 'ecdf'}, 'non admissible aggregate loss distribution provided'
        self.__aggr_loss_dist = value

    @property
    def aggr_loss_dist_method(self):
        return self.__aggr_loss_dist_method

    @aggr_loss_dist_method.setter
    def aggr_loss_dist_method(self, value):
        if value is not None:
            assert value in config.AGGREGATE_LOSS_APPROX_METHOD_LIST, '%r is not one of %s' % (
            value, config.AGGREGATE_LOSS_APPROX_METHOD_LIST)
        self.__aggr_loss_dist_method = value

    @property
    def n_sim(self):
        return self.__n_sim

    @n_sim.setter
    def n_sim(self, value):
        assert isinstance(value, (float, int)), '%r is not an integer' % value
        if isinstance(value, float):
            logger.warning('%r converted to integer' % value)
            value = int(value)
        self.__n_sim = value

    @property
    def random_state(self):
        return self.__random_state

    @random_state.setter
    def random_state(self, value):
        value = int(time.time()) if value is None else value
        assert isinstance(value, int), logger.error('%r is not an integer' % value)
        self.__random_state = value

    @property
    def n_aggr_dist_nodes(self):
        return self.__n_aggr_dist_nodes

    @n_aggr_dist_nodes.setter
    def n_aggr_dist_nodes(self, value):
        assert isinstance(value, (float, int)), '%r is not an integer' % value
        if isinstance(value, float):
            logger.warning('%r converted to integer' % value)
            value = int(value)
        self.__n_aggr_dist_nodes = value

    @property
    def tilt(self):
        return self.__tilt

    @tilt.setter
    def tilt(self, value):
        assert isinstance(value, bool), '%r is not a bool' % value
        self.__tilt = value

    @property
    def tilt_value(self):
        return self.__tilt_value

    @tilt_value.setter
    def tilt_value(self, value):
        assert isinstance(value, (float, int)), '%r is not a float or integer' % value
        self.__tilt_value = value

    @property
    def aggr_n_deductible(self):
        return self.__aggr_n_deductible

    @aggr_n_deductible.setter
    def aggr_n_deductible(self, value):
        assert isinstance(value, (int, float)), 'Aggregate number of deductible value should be an integer or a float'
        assert value >= 0, 'Aggregate number of deductible must be positive'
        self.__aggr_n_deductible = value

    @property
    def n_reinst(self):
        return self.__n_reinst

    @n_reinst.setter
    def n_reinst(self, value):
        assert isinstance(value, (
        int, float)), 'Please provide the number of reinstatements as a positive float or a positive integer'
        assert value >= 0, 'Number of reinstatments must be positive'
        self.__n_reinst = value

    @property
    def reinst_loading(self):
        return self.__reinst_loading

    @reinst_loading.setter
    def reinst_loading(self, value):
        out = 0
        if isinstance(value, (int, float)):
            assert value >= 0, 'The loading for the reinstatement layers must be positive'
            if self.K == float('inf'):
                out = np.repeat(value, 0)
            else:
                out = np.repeat(value, self.K)
        else:
            if isinstance(value, np.ndarray):
                if self.K != float('inf'):
                    assert value.shape[
                               0] == self.K, 'The premium share for the reinstatement layers should be %s dimensional' % str(
                        self.K)
                    assert np.sum(value < 0) == 0, 'The premium share for the reinstatement layers must be positive'
                else:
                    logger.info(
                        'You specified an array for the layers loading which will be disregarded as: \n K=float("inf") is a stop loss contract')
                out = value
        self.__reinst_loading = out

    @property
    def alpha_qs(self):
        return self.__alpha_qs

    @alpha_qs.setter
    def alpha_qs(self, value):
        assert isinstance(value, (float, int)), 'The ceded portion of the quota share must be an integer or a float'
        assert value <= 1 and value > 0, 'The ceded percentage of the quota share must be in (0,1]'
        self.__alpha_qs = value

    @property
    def aggr_cover(self):
        # aggregate cover or limit
        return (self.n_reinst + 1) * self.cover

    @property
    def aggr_deductible(self):
        # aggregate deductible or priority
        return self.aggr_n_deductible * self.deductible

    @property
    def nu(self):
        return 1 - self.sev_model.cdf(self.d)

    @property
    def c(self):
        return self.reinst_loading

    @property
    def K(self):
        return self.n_reinst

    @property
    def L(self):
        return self.aggr_deductible

    @property
    def n(self):
        return self.n_aggr_dist_nodes

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
        Approximate the aggregate loss distribution.
        Calculate aggregate loss empirical pdf, empirical cdf and node values and
        Void method that updates aggr_loss_dist object property.
        If an argument is not provided (``None``) the respective property getter is called.
        If an argument is provided, the respective property setter is called.

        :param aggr_loss_dist_method: computational method to approximate the aggregate loss distribution. One of Fast Fourier Transform ('FFT'), Panjer recursion ('Recursion') and Monte Carlo simulation ('mc').
        :type aggr_loss_dist_method: ``str``, optional
        :param n_aggr_dist_nodes: number of nodes in the approximated aggregate loss distribution.
        :type n_aggr_dist_nodes: ``int``, optional
        :param n_sim: number of simulations of Monte Carlo (mc) method for the aggregate loss distribution approximation.
        :type n_sim: ``int``, optional
        :param random_state: random state for the random number generator in MC.
        :type random_state: ``int``, optional
        :param sev_discr_method: severity discretization method.
        :type sev_discr_method: ``str``, optional
        :param tilt: whether tilting of FFT is present or not (default is 0).
        :type tilt: ``bool``, optional
        :param tilt_value: tilting parameter value of FFT method for the aggregate loss distribution approximation.
        :type tilt_value: ``float``, optional   
        :return: Void
        :rtype: None
        """

        if aggr_loss_dist_method is None and self.aggr_loss_dist_method is None:
            self.aggr_loss_dist = {'nodes': None, 'epdf': None, 'ecdf': None}
            return

        if aggr_loss_dist_method is not None:
            self.aggr_loss_dist_method = aggr_loss_dist_method

        if n_aggr_dist_nodes is not None:
            self.n_aggr_dist_nodes = n_aggr_dist_nodes

        if self.aggr_loss_dist_method == 'mc':
            if n_sim is not None:
                self.n_sim = n_sim
            if random_state is not None:
                self.random_state = random_state
            output = self._mc_simulation()
        elif self.aggr_loss_dist_method == 'recursion':
            if sev_discr_method is not None:
                self.sev_discr_method = sev_discr_method
            output = self._panjer_recursion()
        else:  # self.aggr_loss_dist_method == 'fft'
            if tilt is not None:
                self.tilt = tilt
            if tilt_value is not None:
                self.tilt_value = tilt_value
            if sev_discr_method is not None:
                self.sev_discr_method = sev_discr_method
            output = self._fft()

        self.aggr_loss_dist = output
        return

    def _fft(self):
        """
        Aggregate loss distribution via fast Fourier transform.

        :return: aggregate loss distribution empirical pdf, cdf, nodes
        :rtype: ``dict``

        """
        logger.info('..Approximating aggregate loss distribution via FFT..')

        sevdict = self.severity_discretization()

        fj = sevdict['fj']

        if self.tilt == True:
            tilting_par = 20 / self.n_aggr_dist_nodes if self.tilt_value is None else self.tilt_value
        else:
            tilting_par = 0

        if self.u == float('inf'):
            fj = np.append(fj, np.repeat(0, self.n_aggr_dist_nodes - self.m))
        else:
            fj = np.append(fj, np.repeat(0, self.n_aggr_dist_nodes - self.m - 1))

        f_hat = fft(np.exp(-tilting_par * np.arange(0, self.n_aggr_dist_nodes, step=1)) * fj)
        g_hat = self.fq_model.pgf(f=f_hat)
        g = np.exp(tilting_par * np.arange(0, self.n_aggr_dist_nodes, step=1)) * np.real(ifft(g_hat))

        logger.info('..Distribution via FFT completed..')

        return {'epdf': g,
                'ecdf': np.cumsum(g),
                'nodes': self.h * np.arange(0, self.n_aggr_dist_nodes, step=1)}

    def _panjer_recursion(self):
        """
        Aggregate loss distribution via Panjer recursion.

        :return: aggregate loss distribution empirical pdf, cdf, nodes
        :rtype: ``dict``
        """
        logger.info('..Approximating aggregate loss distribution via Panjer recursion..')

        sevdict = self.severity_discretization()

        fj = sevdict['fj']
        a, b, p0, g = self.abp0g0(fj)

        if self.u == float('inf'):
            fj = np.append(fj, np.repeat(0, self.n_aggr_dist_nodes - self.m))
        else:
            fj = np.append(fj, np.repeat(0, self.n_aggr_dist_nodes - self.m - 1))

        for j in range(1, self.n):
            z = np.arange(1, min(j, len(fj) - 1) + 1)
            g.append((1 / (1 - a * fj[0])) * ((self.fq_model.pmf(1) - (a + b) * p0) * fj[z[-1]] + np.sum(
                ((a + b * z / j) * fj[z] * np.flip(g)[:len(z)]))))

        logger.info('..Panjer recursion completed..')

        return {'epdf': g,
                'ecdf': np.cumsum(g),
                'nodes': self.h * np.arange(0, self.n_aggr_dist_nodes, step=1)}

    def _mc_simulation(self):
        """
        Aggregate loss distribution via Monte Carlo simulation.

        :return: aggregate loss distribution empirical pdf, cdf, nodes.
        :rtype: ``dict``
        """
        logger.info('..Approximating aggregate loss distribution via Monte Carlo simulation..')

        p0 = self.sev_model.cdf(self.d) if self.d > 1e-05 else 0.

        fqsample = self.fq_model.rvs(self.n_sim, random_state=self.random_state)
        svsample = self.sev_model.rvs(int(np.sum(fqsample) + np.ceil(p0 * self.n_sim)), random_state=self.random_state)

        if self.d > 1e-05:
            svsample = svsample[svsample > self.d]
            j = 1
            while svsample.shape[0] < self.n_sim:
                n0 = int(np.ceil(p0 * (self.n_sim - svsample.shape[0])))
                svsample = np.concatenate((svsample, self.sev_model.rvs(n0, random_state=self.random_state + j)))
                svsample = svsample[svsample > self.d]
                j += 1
            svsample = svsample - self.d

        if self.u < float('inf'):
            svsample = np.minimum(svsample, self.u)

        cs = np.cumsum(fqsample).astype(int)

        if fqsample[0:1] == 0:
            xsim = np.array([0])
        else:
            xsim = np.array([np.sum(svsample[0:cs[0]])])

        for i in range(0, self.n_sim - 1):
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

    def aggr_loss_moment(self, central=False, order=1):
        """
        Aggregate loss distribution moment.

        :param central: True if the moment is central, False if the moment is raw.
        :type central: ``bool``
        :param order: order of the moment.
        :type order: ``int``
        :return: emprical moment.
        :rtype: ``numpy.float64``
        """

        self._aggr_loss_dist_check()

        assert isinstance(central, bool), 'The parameter central must be either True or False'
        assert order > 0, 'The parameter order must be a positive integer'
        assert isinstance(order, int), 'The parameter order must be a positive integer'

        if self.aggr_loss_dist_method != 'mc':
            lmmean = np.sum(self.aggr_loss_dist['epdf'] * self.aggr_loss_dist['nodes'])
            return np.sum(self.aggr_loss_dist['epdf'] * ((self.aggr_loss_dist['nodes'] - (central * lmmean)) ** order))
        else:
            return np.mean((self.aggr_loss_dist['nodes'] - central * np.mean(self.aggr_loss_dist['nodes'])) ** order)

    def aggr_loss_ppf(self, q):
        """
        Aggregate loss distribution percent point function, a.k.a. the quantile function, inverse of the cumulative distribution function.

        :param q: vector of probabilities.
        :type q: ``float`` or ``numpy.ndarray``
        :return: vector of quantiles.
        :rtype: ``numpy.ndarray``

        """

        self._aggr_loss_dist_check()

        try:
            if self.aggr_loss_dist_method != 'mc':
                q_ = np.array([])
                q = list(q)
                for i in q:
                    q_ = np.append(q_, [self.aggr_loss_dist['nodes'][self.aggr_loss_dist['ecdf'] >= i][0]])
                return q_
            else:
                return np.quantile(self.aggr_loss_dist['nodes'], q=q)
        except:
            logger.error('Please provide the values for the quantiles in a list')

    def aggr_loss_cdf(self, x):
        """
        Aggregate loss distribution cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``float`` or ``int`` or ``numpy.ndarray``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        self._aggr_loss_dist_check()

        x = np.maximum(x, 0)
        y_ = np.concatenate((0, self.aggr_loss_dist['ecdf']))
        x_ = np.concatenate((0, self.aggr_loss_dist['nodes']))
        output = np.empty(len(x))

        for k in np.arange(len(x)):
            x_temp = x[k]
            j = np.searchsorted(x_, x_temp)
            run = x_[j] - x_[max(j - 1, 0)] if (x_[j] - x_[max(j - 1, 0)]) > 0 else 1
            rise = y_[j] - y_[max(j - 1, 0)]
            output[k] = y_[max(j - 1, 0)] + (x_temp - x_[max(j - 1, 0)]) * rise / run

        return output

    def aggr_loss_rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``
        """
        self._aggr_loss_dist_check()

        random_state = int(time.time()) if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        np.random.seed(random_state)

        output = np.random.choice(self.aggr_loss_dist['nodes'], size=size, p=self.aggr_loss_dist['epdf'])

        return output

    def aggr_loss_mean(self):
        return self.aggr_loss_moment(central=False, order=1)

    def aggr_loss_std(self):
        return self.aggr_loss_moment(central=True, order=2) ** 1 / 2

    def aggr_loss_skewness(self):
        return self.aggr_loss_moment(central=True, order=3) / self.aggr_loss_moment(central=True, order=2) ** 3 / 2

    def _stop_loss_pricing(self, t):
        """
        Expected value of a stop loss contract with deductible t, also referred to as retention or priority.

        :param t: deductible.
        :type t: ``float``
        :return: stop loss contract expected value.

        """
        t = np.repeat([t], 1)
        x = self.aggr_loss_dist['nodes']
        if self.aggr_loss_dist_method == 'mc':
            if t.shape[0] == 1:
                return (np.mean(np.maximum(x - t, 0)))
            else:
                lg = t.shape[0]
                obs = x.shape[0]
                v1_ = np.tile(x, lg).reshape(lg, -1) - np.repeat(t, obs).reshape(lg, -1)
                v1_[v1_ < 1e-6] = .0
                return (np.apply_along_axis(arr=v1_, func1d=np.mean, axis=1))
        else:
            probs = self.aggr_loss_dist['epdf']
            if t.shape[0] == 1:
                return (np.sum(np.maximum(x - t, 0) * probs))
            else:
                lg = t.shape[0]
                obs = x.shape[0]
                v1_ = np.tile(x, lg).reshape(lg, -1) - np.repeat(t, obs).reshape(lg, -1)
                v1_[v1_ < 1e-6] = .0
                v2_ = v1_ * np.tile(probs, lg).reshape(lg, -1)
                return (np.apply_along_axis(arr=v2_, func1d=np.sum, axis=1))

    def _reinstatement_pricing(self):
        """
        Reinstatements pricing.

        :return: final reinstatements pricing.
        :rtupe: ``numpy.ndarray``
        """
        output = self._stop_loss_pricing(self.L) - self._stop_loss_pricing(self.L + (self.K + 1) * self.u)
        if self.K > 0:
            lower_k = np.linspace(start=0, stop=self.K, num=self.K + 1)
            dLk = (self._stop_loss_pricing(self.L + lower_k[:-1] * self.u) - self._stop_loss_pricing(
                self.L + lower_k[1:] * self.u))
            den = 1 + np.sum(dLk * self.c) / self.u
            output = output / den
        return output

    def pricing(self):
        """
        Actuarial pricing (also referred to as costing) for proportional and non-proportional reinsurance contracts, such as
        quota share, excess-of-loss, excess-of-loss with reinstatements, stop loss.

        :return: contract actuarial pricing (also referred to as costing).
        """

        p_ = None
        if self.K == float('inf'):
            p_ = self._stop_loss_pricing(t=self.L)

        if self.K != float('inf'):
            p_ = self._reinstatement_pricing()

        data = [
            ['Deductible', 'd', self.d],
            ['Cover', 'u - d', self.u - self.d],
            ['Upper priority', 'u', self.u],
            ['Aggregate priority', 'L', self.L],
            ['Quota share ceded portion', 'alpha', self.alpha_qs]
        ]
        if self.K != float('inf'):
            data.extend([['Number of reinstatements', 'K', self.K]])

        data.append(['Pure premium', 'P', self.alpha_qs * p_])
        print('{: >20} {: >20} {: >20} {: >20}'.format(' ', *['Contract specification', 'parameter', 'value']))
        print('{: >20} {: >20}'.format(' ', *[' ==================================================================']))
        for row in data:
            print('{: >20} {: >20} {: >20} {: >20}'.format('', *row))
        print('\n Reinstatement layer loading c: ', self.c)
        if self.aggr_loss_dist_method == 'mc':
            print(self.aggr_loss_dist_method, '\t n_sim: ', self.n_sim, '\t random_state:', self.random_state)
        else:
            print(self.aggr_loss_dist_method, '\t n_sev_discr_nodes m: ', self.m, '\t n_aggr_dist_nodes n: ', self.n)

    def print_aggr_loss_specs(self):
        """
        Print aggregate loss distribution approximation specifications.
        :return: Void
        :rtype: None
        """

        data = [
            ['aggr_loss_dist_method', self.aggr_loss_dist_method],
            ['n_aggr_dist_nodes (n)', self.n_aggr_dist_nodes],
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

        data.extend([
            ['sev_discr_method', self.sev_discr_method],
            ['sev_discr_step', self.sev_discr_step],
            ['n_sev_discr_nodes', self.n_sev_discr_nodes]
        ])

        print('{: >20} {: >20} {: >20} {: >20}'.format(' ', ' ', *['Aggregate loss feature', 'value']))
        print('{: >20} {: >20}'.format(' ', *[' ==================================================================']))
        for row in data:
            print('{: >20} {: >20} {: >20} {: >20}'.format('', *row))
        return

    def print_contract_specs(self):
        """
        Print contract specifications.
        :return: Void
        :rtype: None
        """
        data = [
            ['deductible', self.deductible],
            ['cover', self.cover],
            ['aggr_deductible', self.aggr_deductible],
            ['alpha_qs', self.alpha_qs],
            ['reinst_loading', self.reinst_loading],
            ['n_reinst', self.n_reinst],
        ]

        print('{: >20} {: >20} {: >20} {: >20}'.format(' ', ' ', *['Contract specification', 'value']))
        print('{: >20} {: >20}'.format(' ', *[' ==================================================================']))
        for row in data:
            print('{: >20} {: >20} {: >20} {: >20}'.format('', *row))

        return

    def _aggr_loss_dist_check(self):
        """
        Assert whether the aggregate loss distribution is not missing.
        Helper method called before executing other methods on ``aggr_loss_dist`` property.
        
        :return: Void
        :rtype: None
        """
        if self.aggr_loss_dist is None:
            logger.error(
                'Aggregate loss distribution missing, use aggr_loss_dist_calculate method first'
            )
