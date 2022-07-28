import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from . import helperfunctions as hf

from twiggy import quick_setup, log

quick_setup()
logger = log.name('loss_reserve')


class LossReserve:
    """
    Class to compute the loss reserve.
    The available models are the deterministic Fisher-Lange and the collective risk model.
    Input company data must be ``numpy.ndarray`` data on numbers and payments must be in triangular form:
    two-dimensional ``numpy.ndarray`` with shape (I,J) with I=J.

    :param tail: set it to True when the tail estimate is required. Default False.
    :type tail: ``bool``
    :param reserving_method: one of the reserving methods supported by the GemAct package.
    :type reserving_method: ``str``
    :param claims_inflation: claims inflation. In case no tail is present and the triangular data IxJ matrices,
                            claims_inflation must be J-1 dimensional. When a tail estimate is required, it must be
                            J dimensional. In case no tail is present it must be J-1 dimensional.
    :type claims_inflation: ``numpy.ndarray``
    :param czj: severity coefficient of variation by development year. It is set to None in case the crm is selected as
                reserving method. When a tail estimate is required, it must be J dimensional.
                In case no tail is present it must be J-1 dimensional.
    :type czj: ``numpy.ndarray``

    :param ntr_sim: Number of simulated triangles in the c.r.m reserving method.
    :type ntr_sim: ``int``
    :param set_seed: Simulation seed to make the c.r.m reserving method results reproducible.
    :type set_seed: ``int``
    :param mixing_fq_par: Mixing frequency parameters.
    :type mixing_fq_par: ``dict``
    :param mixing_sev_par: Mixing severity parameters.
    :type mixing_sev_par: ``dict``
    :param custom_alphas: optional, custom values for the alpha parameters.
    :type custom_alphas: ``numpy.ndarray``
    :param custom_ss: optional, custom values for the settlement speed.
    :type custom_ss: ``numpy.ndarray``

    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *ip_tr* = (``numpy.ndarray``) --
            Incremental payments' triangle
        * *cumulative_tr* = (``numpy.ndarray``) --
            Cumulative payments' triangle
        * *cp_tr* = (``numpy.ndarray``) --
            Cased payments triangle
        * *in_tr* = (``numpy.ndarray``) --
            Incurred number
        * *cn_tr* = (``numpy.ndarray``) --
            Cased number

    """

    def __init__(self, tail=False,
                 claims_inflation=None,
                 custom_alphas=None,
                 custom_ss=None,
                 reserving_method="fisher_lange",
                 ntr_sim=1000,
                 mixing_fq_par=None,
                 czj=None,
                 set_seed=42,
                 cumulative_tr=None,
                 mixing_sev_par=None, **kwargs):

        self.tail = tail
        self.reserving_method = reserving_method

        try:
            self.j = kwargs['incremental_payments'].shape[1]
            self.ix = np.tile(np.arange(0, self.j), self.j).reshape(self.j,
                                                                    self.j) + np.tile(np.arange(1, self.j + 1),
                                                                                      self.j).reshape(self.j,
                                                                                                      self.j).T
        except Exception:
            logger.error('The Incremental Payments triangle must be a two dimensional '
                         'numpy.ndarray with same shape on both dimensions')
            raise

        # 1d-arrays
        self.claims_inflation = claims_inflation
        self.reported_claims = kwargs['reported_claims']
        self.czj = czj
        self.czj_t = np.tile(self.czj, self.j).reshape(self.j, -1)

        # triangles
        self.ip_tr = kwargs['incremental_payments']
        self.cp_tr = kwargs['cased_payments']
        self.in_tr = kwargs['incurred_number']
        self.cn_tr = kwargs['cased_number']
        self.ap_tr = None
        self.cumulative_tr = cumulative_tr

        # variables and other objects
        if self.reserving_method == 'crm':
            self.ntr_sim = ntr_sim
            self.set_seed = set_seed
            self.mixing_fq_par = mixing_fq_par
            self.mixing_sev_par = mixing_sev_par
            self.gamma1 = stats.gamma(**self.mixing_fq_par)
            self.gamma2 = stats.gamma(**self.mixing_sev_par)
            self.gamma3 = stats.gamma
            self.pois = stats.poisson

        # attributes with opportunities of not standard customization
        if custom_alphas is not None:
            self.alpha_fl = custom_alphas
        else:
            self.alpha_fl = self._alpha_computer()

        if custom_ss is not None:
            self.ss_fl_ = custom_alphas
        else:
            self.ss_fl_ = self._ss_computer()

        # results
        self.ss_tr = self._ss_triangle()  # triangle of settlement speed
        self.predicted_i_numbers = self._fill_numbers()  # individual payments triangle

        self.fl_reserve = self._fl_reserve()  # fisher-lange reserve

        if self.reserving_method == 'crm':
            self.crm_reserve, self.m_sep, self.skewness = self._stochastic_crm()
            # crm reserving: value of the reserve, mean-squared error of prediction and skewness.

    @property
    def tail(self):
        return self.__tail

    @tail.setter
    def tail(self, var):
        assert isinstance(var, bool), logger.error("tail must be a boolean")
        self.__tail = var

    @property
    def reserving_method(self):
        return self.__reserving_method

    @reserving_method.setter
    def reserving_method(self, var):
        assert var.lower() in ['chl', 'fisher_lange', 'crm'], logger.error("%r is not supported from our package" % var)
        assert isinstance(var, str), logger.error("%r must be a string" % var)
        if not var.lower() == var:
            logger.warning("What did you mean with %r? 'reserving_method' is set to %r." % (var, var.lower()))
        self.__reserving_method = var.lower()

    # 1-d arrays
    @property
    def claims_inflation(self):
        return self.__claims_inflation

    @claims_inflation.setter
    def claims_inflation(self, var):
        if var is None:
            self.__claims_inflation = np.repeat(1, self.j)
        else:
            try:
                var = np.array(var, dtype=float)
            except Exception:
                logger.error("Provide a numpy.ndarray for claims inflation")
                raise

            if self.tail:
                assert var.shape[0] == self.j, logger.error(
                    "The vector of inflation you provided is not correctly shaped. \n Length must be %s" % self.j)
            else:
                assert var.shape[0] == self.j - 1, logger.error(
                    "The vector of inflation you provided is not correctly shaped. \n Length must be %s" % self.j - 1)

            self.__claims_inflation = var

    @property
    def reported_claims(self):
        return self.__reported_claims

    @reported_claims.setter
    def reported_claims(self, var):
        var = np.array(var)
        assert var.shape[0] == self.j, logger.error(
            'Reported Claims must be provided as a \n 1-d array with length %s' % self.j)
        logger.info(
            "A correct 1-d vector for Reported claims was provided. "
            "\n Make sure the reported claims on the last position corresponds to the most recent data")
        self.__reported_claims = var

    @property
    def czj(self):
        return self.__czj

    @czj.setter
    def czj(self, var):
        if self.reserving_method in ['chl', 'fisher_lange']:
            self.__czj = None
        else:
            if var is None:
                self.__czj = np.repeat(.001, self.j + self.tail)
            else:
                try:
                    var = np.array(var, dtype=float)
                except Exception:
                    logger.error("Provide a numpy.ndarray for czj")
                    raise

                if self.tail:
                    assert var.shape[0] == self.j, logger.error(
                        "The vector of czj you provided is not correctly shaped. \n Length must be %s" % self.j)
                    var = np.concatenate(([.0], var))
                else:
                    assert var.shape[0] == self.j - 1, logger.error(
                        "The vector of czj you provided is not correctly shaped. \n Length must be %s" % self.j - 1)
                    var = np.concatenate(([.0], var))

                self.__czj = var

    # Triangles
    @property
    def ip_tr(self):
        return self.__ip_tr

    @ip_tr.setter
    def ip_tr(self, var):
        if self.reserving_method in ['chl']:
            if var is None:
                self.__ip_tr = var
            else:
                var = np.array(var).astype(float)
                assert type(var) == np.ndarray, logger.error(
                    'The Incremental Payments triangle must be a two dimensional numpy.ndarray')
                assert var.shape[0] == var.shape[1], logger.error(
                    'The Incremental Payments triangle must be a two dimensional with same shape on both dimensions. '
                    '\n The triangle shape is: %s' % str(
                        var.shape))

                nans = np.isnan(
                        var)

                if np.sum(nans) > 0:
                    assert np.min(self.ix[nans]) > self.j, logger.error(
                        'Check your Incremental Payments input.\n Not valid values in the upper triangle.')
                self.__ip_tr = var
        else:
            var = np.array(var).astype(float)
            assert type(var) == np.ndarray, logger.error(
                'The Incremental Payments triangle must be a two dimensional numpy.ndarray')
            assert var.shape[0] == var.shape[1], logger.error(
                'The Incremental Payments triangle must be a two dimensional with same shape on both dimensions. '
                '\n The triangle shape is: %s' % str(
                    var.shape))

            nans = np.isnan(
                var)
            if np.sum(nans) > 0:
                assert np.min(self.ix[nans]) > self.j, logger.error(
                    'Check your Incremental Payments input.\n Not valid values in the upper triangle.')
            self.__ip_tr = var

    @property
    def cumulative_tr(self):
        return self.__cumulative_tr

    @cumulative_tr.setter
    def cumulative_tr(self, var):
        if self.reserving_method in ['chl']:
            if self.ip_tr is None:
                var = np.array(var).astype(float)
                assert type(var) == np.ndarray, logger.error(
                    'The Incremental Payments triangle must be a two dimensional numpy.ndarray')
                assert var.shape[0] == var.shape[1], logger.error(
                    'The Incremental Payments triangle must be a two dimensional with same shape on both dimensions. '
                    '\n The triangle shape is: %s' % str(
                        var.shape))

                nans = np.isnan(
                    var)
                if np.sum(nans) > 0:
                    assert np.min(self.ix[nans]) > self.j, logger.error(
                        'Check your Incremental Payments input.\n Not valid values in the upper triangle.')
                self.__cumulative_tr = var
            else:
                cumulative_ = self.ip_tr.copy()
                for i in range(1, cumulative_.shape[1]):
                    cumulative_[:, i] = cumulative_[:, i] + cumulative_[:, i - 1]
                self.__cumulative_tr = cumulative_
        else:
            self.__cumulative_tr = None

    @property
    def cp_tr(self):
        return self.__cp_tr

    @cp_tr.setter
    def cp_tr(self, var):
        if self.reserving_method in ['fisher_lange', 'crm']:
            var = np.array(var).astype(float)
            assert type(var) == np.ndarray, logger.error(
                'The Cased Payments triangle must be a two dimensional numpy.ndarray')
            assert var.shape[0] == var.shape[1], logger.error(
                'The Cased Payments triangle must be a two dimensional with same shape on both dimensions. '
                '\n The triangle shape is: %s' % str(
                    var.shape))

            nans = np.isnan(
                var)
            if np.sum(nans) > 0:
                assert np.min(self.ix[nans]) > self.j, logger.error(
                    'Check your Cased Payments input.\n Not valid values in the upper triangle.')
            self.__cp_tr = var
        else:
            self.__cp_tr = var

    @property
    def in_tr(self):
        return self.__in_tr

    @in_tr.setter
    def in_tr(self, var):
        if self.reserving_method in ['fisher_lange', 'crm']:
            var = np.array(var).astype(float)
            assert type(var) == np.ndarray, logger.error(
                'The Incurred Number triangle must be a two dimensional numpy.ndarray')
            assert var.shape[0] == var.shape[1], logger.error(
                'The Incurred Number triangle must be a two dimensional with same shape on both dimensions. '
                '\n The triangle shape is: %s' % str(
                    var.shape))

            nans = np.isnan(
                var)

            if np.sum(nans) > 0:
                assert np.min(self.ix[nans]) > self.j, logger.error(
                    'Check your Incurred Number input. \n Not valid values in the upper triangle.')
            self.__in_tr = var
        else:
            self.__in_tr = None

    @property
    def cn_tr(self):
        return self.__cn_tr

    @cn_tr.setter
    def cn_tr(self, var):
        if self.reserving_method in ['fisher_lange', 'crm']:
            var = np.array(var).astype(float)
            assert type(var) == np.ndarray, logger.error(
                'The Cased Number triangle must be a two dimensional numpy.ndarray')
            assert var.shape[0] == var.shape[1], logger.error(
                'The Cased Number triangle must be a two dimensional with same shape on both dimensions. '
                '\n The triangle shape is: %s' % str(
                    var.shape))

            nans = np.isnan(
                var)

            if np.sum(nans) > 0:
                assert np.min(self.ix[nans]) > self.j, logger.error(
                    'Check your Cased Number input.\n Not valid values in the upper triangle.')
            self.__cn_tr = var
        else:
            self.__cn_tr = None

    @property
    def ntr_sim(self):
        return self.__ntr_sim

    @ntr_sim.setter
    def ntr_sim(self, var):
        try:
            var = int(var)
            assert isinstance(var, int), logger.error(
                "The number of simulated triangles for the crm reserving model must be provided as an integer")
            self.__ntr_sim = var
        except Exception:
            logger.error("Please provide the number of simulated triangles for the crm reserving as an integer")
            raise

    @property
    def set_seed(self):
        return self.__set_seed

    @set_seed.setter
    def set_seed(self, var):
        try:
            var = int(var)
            assert isinstance(var, int), logger.error(
                "The seed number for the crm reserving model must be provided as an integer")
            self.__set_seed = var
        except Exception:
            logger.error("Please provide the seed number for the crm reserving as an integer")
            raise

    @property
    def mixing_fq_par(self):
        return self.__mixing_fq_par

    @mixing_fq_par.setter
    def mixing_fq_par(self, var):
        if self.reserving_method == "crm":
            try:
                assert isinstance(var, dict), logger.error(
                    "The frequency mixing parameters for the crm reserving model must be provided as a dictionary")
                assert all(item in list(var.keys()) for item in ["scale", "a"]), logger.error(
                    "The mixing frequency parameters for the crm reserving model must be provided as 'a' and 'scale'. "
                    "See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html")
                self.__mixing_fq_par = var
            except Exception:
                logger.error("Please provide the mixing frequency parameters for the crm reserving as a dictionary")
                raise
        else:
            self.__mixing_fq_par = var

    @property
    def mixing_sev_par(self):
        return self.__mixing_sev_par

    @mixing_sev_par.setter
    def mixing_sev_par(self, var):
        if self.reserving_method == "crm":
            try:
                assert isinstance(var, dict), logger.error(
                    "The severity mixing parameters for the crm reserving model must be provided as a dictionary")
                assert all(item in list(var.keys()) for item in ["scale", "a"]), logger.error(
                    "The mixing severity parameters for the crm reserving model must be provided as 'a' and 'scale'. "
                    "See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html")
                self.__mixing_sev_par = var
            except Exception:
                logger.error("Please provide the mixing frequency parameters for the crm reserving as a dictionary")
                raise
        else:
            self.__mixing_sev_par = var

    @property
    def ap_tr(self):
        return self.__ap_tr

    @ap_tr.setter
    def ap_tr(self, var):

        if var is None:
            temp = self.ip_tr / self.in_tr
            n = self.j - 1
            for j in range(1, self.j):
                temp[n - j + 1:, j] = temp[n - j, j] * np.cumprod(self.claims_inflation[:j])

            if self.tail:
                self.__ap_tr = np.column_stack((temp, temp[0, -1] * np.cumprod(self.claims_inflation)))
            else:
                self.__ap_tr = temp

    # attributes with not-standard customization
    @property
    def alpha_fl(self):
        return self.__alpha_fl

    @alpha_fl.setter
    def alpha_fl(self, var):
        assert var.shape[0] == self.j + self.tail - 1, logger.error(
            "The alpha vector length must be %s" % (self.j + self.tail - 1))
        self.__alpha_fl = np.array(var)

    @property
    def ss_fl_(self):
        return self.__ss_fl_

    @ss_fl_.setter
    def ss_fl_(self, var):
        var = np.array(var)
        assert var.shape[0] == self.j + self.tail - 1, logger.error(
            "The alpha vector length must be %s" % (self.j + self.tail - 1))
        assert np.abs(np.sum(var) - 1) < 1e+04, logger.error("The Settlement Speed vector must sum to one")
        self.__ss_fl_ = var

    # methods
    def _alpha_computer(self):
        """
        It computes Fisher-Lange alpha. Given a JxJ triangle, this is going to be
        J-1 dimensional in case no tail is present and J dimensional in case of tail estimates.

        :return: vectors of alpha
        :rtype: ``numpy.ndarray``
        """
        temp_in_ = self.in_tr.copy()
        temp_cn_ = self.cn_tr.copy()
        temp_in_[self.ix > self.j] = 0.
        temp_cn_[self.ix > self.j] = np.nan
        v_ = np.flip(np.apply_along_axis(arr=temp_in_, func1d=np.sum, axis=1))
        # create a matrix with the cumulative number of claims in the diagonal
        tot_inc = np.rot90(np.array([v_, ] * self.j))
        # create a matrix with cumulative sums
        cum_inc = np.apply_along_axis(arr=temp_in_, func1d=np.cumsum, axis=1)
        # create a matrix
        dg_ = np.flip(np.diag(np.rot90(temp_cn_)))
        mx2_ = np.rot90(np.array([dg_, ] * self.j))
        # create the matrix of claims incurred in the future years
        mx1_ = (tot_inc - cum_inc + mx2_) / temp_cn_

        mx1_[self.ix >= self.j] = np.nan

        if not self.tail:
            return np.apply_along_axis(arr=mx1_[:, :-1], func1d=np.nanmean, axis=0)
        else:
            return np.concatenate((np.apply_along_axis(arr=mx1_[:, :-1], func1d=np.nanmean, axis=0), [1.]))

    def _ss_computer(self):
        """
        It computes Fisher-Lange settlement speeds. Given a JxJ triangle, this is going to be
        J-1 dimensional in case no tail is present and J dimensional in case of tail estimates.

        :return: settlement speed
        :rtype: ``numpy.ndarray``
        """
        temp = np.flip((np.diag(np.rot90(self.in_tr)) * self.reported_claims[-1] / self.reported_claims)[:-1])

        if self.tail:
            temp = np.concatenate((temp, [self.cn_tr[0, -1] * self.reported_claims[-1] / self.reported_claims[0]]))

        return temp / np.sum(temp)

    def _ss_triangle(self):
        """
        It turns Fisher-Lange settlement speed array into a triangle. Given a JxJ triangle, this is going to be
        JxJ-1 dimensional in case no tail is present and JxJ dimensional in case of tail estimates.

        :return: settlement speed triangle
        :rtype: ``numpy.ndarray``
        """
        mx1_ = np.array([np.concatenate(([0.], self.ss_fl_)), ] * self.j)
        if self.tail:
            new_ix = np.concatenate((self.ix <= self.j, np.repeat(False, self.j).reshape(self.j, 1)), axis=1)
            mx1_[new_ix] = np.nan
        else:
            mx1_[self.ix <= self.j] = np.nan

        return np.apply_along_axis(arr=mx1_, func1d=hf.normalizernans, axis=1)

    def _fill_numbers(self):
        """
        It computes the lower triangle of numbers. Given a JxJ triangle, this is going to be
        JxJ dimensional in case no tail is present and Jx(J+1) dimensional in case of tail estimates.

        :return: number of payments
        :rtype: ``numpy.ndarray``
        """
        # quotas
        if not self.tail:
            v_ = np.concatenate((self.alpha_fl, [1.]))
        else:
            v_ = self.alpha_fl
        alq_ = np.flip(np.array([v_, ] * self.j).T)

        # diagonal
        dg_ = np.diag(np.flip(np.rot90(self.cn_tr)))
        amounts_ = np.flip(np.array([dg_, ] * self.j).T)

        if self.tail:
            alq_ = np.column_stack((np.ones(self.j), alq_))
            amounts_ = np.column_stack((np.ones(self.j), amounts_))

        # final development
        ss_amounts_alq = self.ss_tr * amounts_ * alq_

        if self.tail:
            new_ix = np.concatenate((self.ix > self.j, np.repeat(True, self.j).reshape(self.j, 1)), axis=1)
            temp_ = np.column_stack((self.in_tr, np.ones(self.j)))
            temp_[new_ix] = ss_amounts_alq[~np.isnan(ss_amounts_alq)]
            return temp_
        else:
            temp_ = self.in_tr.copy()
            temp_[self.ix > self.j] = ss_amounts_alq[~np.isnan(ss_amounts_alq)]
            return temp_

    def _fl_reserve(self):
        """
        It computes the reserve via the Fisher-Lange reserving model.

        :return: fisher-lange reserve
        :rtype: ``numpy.float64``
        """
        # self.predicted_i_numbers=self._fill_numbers()
        self.predicted_i_payments = self.predicted_i_numbers * self.ap_tr

        if self.tail:
            new_ix = np.concatenate((self.ix > self.j, np.repeat(True, self.j).reshape(self.j, 1)), axis=1)
            self.predicted_i_payments[0, -1] = self.cp_tr[0, -1]
            return np.sum(self.predicted_i_payments[new_ix])
        else:
            return np.sum(self.predicted_i_payments[self.ix > self.j])

    def _stochastic_crm(self):
        """
        It computes the reserve according to the collective risk model based on the Fisher-Lange.

        :return: reserve prediction (simulations mean), reserve m_sep prediction, reserve skewness
        :rtype:``numpy.float64``,``numpy.float64``,``numpy.float64``

        """
        flag_ = np.repeat('ay' + str(0), self.ap_tr.shape[1])  # create a flag that will be used to pick the correct ay
        for ay in range(1, self.ap_tr.shape[0]):
            cell_ = 'ay' + str(ay)
            temp_ = np.repeat(cell_, self.ap_tr.shape[1])
            flag_ = np.vstack((flag_, temp_))

        if not self.tail:

            v1_ = self.predicted_i_numbers[self.ix > self.j]  # numbers lower triangle
            v2_ = self.ap_tr[self.ix > self.j]  # average payments lower triangle
            czj_v = self.czj_t[self.ix > self.j]  # coefficient of variation lower triangle
            flag_v = flag_[self.ix > self.j]
        else:
            new_ix = np.concatenate((self.ix > self.j, np.repeat(True, self.j).reshape(self.j, 1)),
                                    axis=1)  # lower triangle and tail
            v1_ = self.predicted_i_numbers[new_ix]  # numbers lower triangle and tail
            v2_ = self.ap_tr[new_ix]  # average payments lower triangle and tail
            czj_v = self.czj_t[new_ix]  # coefficient of variation lower triangle and tail
            flag_v = flag_[new_ix]

        np.random.seed(self.set_seed)
        output = np.array([])
        # now = datetime.now()
        self.crm_sep_ay = np.array([], dtype=np.float64)  # to store the mean squared error of prediction
        self.crm_ul_ay = np.array([], dtype=np.float64)  # to store the ultimate cost
        self.ay_reserve_crm = np.array([], dtype=np.float64)  # to store the reserve by time period
        mean_squared_ep_temp_ = np.array([], dtype=np.float64)
        ultimate_temp = np.array([], dtype=np.float64)
        for i in range(0, len(v1_)):  # crm computed on each lower triangle cell

            f_ = flag_v[i]  # flag the cell with the correspondent ay

            if i+1 < len(flag_v):
                fp_ = flag_v[
                    i + 1]
            else:
                fp_ = 'stop'

            p1_ = v1_[i]  # cell numbers
            p2_ = v2_[i]  # cell average payment
            p3_ = czj_v[i]  # cell coefficient of variation

            vec1_ = p1_ * self.gamma1.rvs(self.ntr_sim)
            vec2_ = p2_ ** 2 / (p3_ * self.gamma2.rvs(self.ntr_sim))
            vec3_ = p3_ * self.gamma2.rvs(self.ntr_sim) / p2_

            vec4_ = np.apply_along_axis(func1d=hf.lrcrm_f1, arr=vec1_.reshape(-1, 1), axis=1, dist=self.pois).reshape(
                -1, )  # simulate all the CRMs for the cell
            mx_ = np.array([vec4_, vec2_, vec3_]).T  # create a matrix of parameters
            temp_ = np.apply_along_axis(axis=1, arr=mx_, func1d=hf.lrcrm_f2,
                                        dist=self.gamma3)  # simulate the reserves
            if i == 0:
                output = np.array(temp_).reshape(-1, 1)
            else:
                output = np.column_stack((output, np.array(temp_).reshape(-1, 1)))

            mean_squared_ep_temp_ = np.concatenate(
                (mean_squared_ep_temp_, temp_))  # add to the ay estimate the simulated CRMs. It will be used for the
            # mean reserve as well
            ultimate_temp = np.concatenate((ultimate_temp, [np.mean(temp_)]))
            if f_ != fp_:  # in case next cell belongs to another ay, reserve variability is computed
                self.crm_sep_ay = np.concatenate((self.crm_sep_ay, [np.std(mean_squared_ep_temp_)]))
                self.ay_reserve_crm = np.concatenate((self.ay_reserve_crm, [np.sum(ultimate_temp)]))
                self.crm_ul_ay = np.concatenate((self.crm_ul_ay, [np.cumsum(ultimate_temp)[-1]]))
                mean_squared_ep_temp_ = np.array([], dtype=np.float64)
                ultimate_temp = np.array([], dtype=np.float64)

            sys.stdout.write("\r")
            v = round((i + 1) / len(v1_) * 100, 3)
            str1 = ("[%-" + str(len(v1_)) + "s] %d%%")
            sys.stdout.write(str1 % ('=' * i, v))
            sys.stdout.flush()

        # then = datetime.now()
        print("")
        # logger.info('Time elapsed %s' % str(then - now))
        if not self.tail:
            self.ay_reserve_crm = np.concatenate(([0], self.ay_reserve_crm))
            self.crm_sep_ay = np.concatenate(([0], self.crm_sep_ay))
            self.crm_ul_ay = np.concatenate(([0], self.crm_ul_ay))
        else:  # I need to fill the last line of the ultimate in advance in case of CRM
            diagonal_cml_ = self.predicted_i_payments[-1, 0]
            self.crm_ul_ay[self.predicted_i_payments.shape[0] - 1] = self.crm_ul_ay[
                                                                self.predicted_i_payments.shape[0] - 1] + diagonal_cml_

        for ay in range(0, self.predicted_i_payments.shape[
                               0] - self.tail):
            # add to the estimated random cumulative payments the upper triangle amounts
            diagonal_cml_ = np.cumsum(self.predicted_i_payments[ay, :(self.j - ay - 1 - self.tail + 1)])[
                -1]  # separate it to make it readable
            self.crm_ul_ay[ay] = self.crm_ul_ay[ay] + diagonal_cml_

        reserves_ = np.apply_along_axis(arr=output, func1d=np.sum, axis=1)
        return np.mean(reserves_), np.std(reserves_), stats.skew(reserves_)

    def ss_plot(self, start_=0):
        """
        It plots the settlement speed vector for each accident year.
        :param start_: starting accident year from which to plot.
        :type start_: ``int``
        """
        x_ = np.arange(0, self.j + self.tail)
        plt.title('Plot of the settlement speed from accident year %s' % start_)
        plt.xlabel('Development Year')
        plt.ylabel('Settlement Speed')
        for i in range(start_, self.j):
            plt.plot(x_, self.ss_tr[i, :], '-.', label='AY %s' % i)
            plt.legend()
        plt.show()

    def average_cost_plot(self):
        """
        It plots the mean average cost for each development year.
        """
        x_ = np.arange(0, self.j + self.tail)
        plt.title('Plot of the Average Cost (mean of each DY, data and predictions)')
        plt.xlabel('Development Year')
        plt.ylabel('Average Cost')
        y_ = np.apply_along_axis(arr=self.ap_tr, func1d=np.mean, axis=0)
        plt.plot(x_, y_, '-.', label='Mean Average Cost')
        plt.show()

    def alpha_plot(self):
        """
        It plots the Fisher-Lange alpha.
        """
        x_ = np.arange(0, self.j + self.tail - 1)
        plt.title('Plot of Alpha')
        plt.plot(x_, self.alpha_fl, '-.', label='Alpha')
        plt.xlabel('Development Year')
        plt.ylabel('Alpha')
        plt.show()

    def _reserve_by_ay_fl(self):
        """
        It computes the Fisher-Lange reserve for each accident year and
        the Fisher-Lange ultimate cost for each accident year.

        :return: reserve for each accident year,ultimate cost for each accident year
        :rtype: ``numpy.ndarray``, ``numpy.ndarray``
        """
        self.ay_reserveFL = np.array([])
        self.ay_ultimateFL = np.array([])
        for ay in range(0, self.predicted_i_payments.shape[0]):
            v_ = self.predicted_i_payments[ay, :]
            self.ay_reserveFL = np.concatenate((self.ay_reserveFL, [np.sum(v_[(self.j - ay):])]))
            self.ay_ultimateFL = np.concatenate((self.ay_ultimateFL, [np.cumsum(v_)[-1]]))

    def claims_reserving(self):
        """
        Table with claims reserve results.
        When the stochastic reserve according to the collective risk model is computed the results
        are compared with the Fisher-Lange.

        """
        self._reserve_by_ay_fl()
        ay_ = np.arange(0, self.predicted_i_payments.shape[0])
        data = np.dstack((ay_, self.ay_ultimateFL, self.ay_reserveFL)).reshape(-1, 3)
        if self.reserving_method == 'crm':
            data2 = np.dstack((self.crm_ul_ay, self.ay_reserve_crm, self.crm_sep_ay)).reshape(-1, 3)
            data = np.column_stack((data, data2))
        l_ = ['time', 'ultimate FL', 'reserve FL']
        if self.reserving_method == 'crm':
            l2_ = ['ultimate CRM', 'reserve CRM', 'm_sep CRM']
            l_.extend(l2_)
        s_ = "{: >20} {: >20} {: >20} {: >20}"
        if self.reserving_method == 'crm':
            s_ = s_ + " {: >20} {: >20} {: >20}"
        print(s_.format(" ", *l_))
        print("{: >20} {: >20}".format(" ", *[
            " ================================================================"
            "===================================================================="]))
        for row in data:
            print(s_.format("", *row))
        print('\n FL reserve: ', self.fl_reserve)
        if self.reserving_method == 'crm':
            print('\n CRM reserve: ', self.crm_reserve)
            print('\n CRM m_sep: ', self.m_sep)
