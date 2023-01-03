from .libraries import *
from . import config
from . import helperfunctions as hf

quick_setup()
logger = log.name('lossreserve')

class AggregateData:
    """
    Triangular data sets.

    """
    def __init__(self,
                 incremental_payments=None,
                 cased_payments=None,
                 incurred_number=None,
                 cased_number=None,
                 reported_claims=None
                 ):

        # triangles
        ## check model parametrization on the incremental payments
        self.ip_tr = incremental_payments
        self.cp_tr = cased_payments
        self.in_tr = incurred_number
        self.cn_tr = cased_number
        self.j = None
        self.ix = None
        self.reported_claims = reported_claims

    @property
    def ip_tr(self):
        return self.__ip_tr

    @ip_tr.setter
    def ip_tr(self, var):
        name = 'ip_tr'
        var = hf.ndarray_try_convert(var, name, logger, type=float)
        hf.check_condition(var.shape[0], var.shape[1], name, logger)
        nans = np.isnan(var)
        j = var.shape[1]
        ix = np.tile(np.arange(0, j), j).reshape(j, j) + np.tile(np.arange(1, j + 1), j).reshape(j, j).T
        if np.sum(nans) > 0:
            assert np.min(ix[nans]) > j, logger.error(
                'Not valid values in %s upper triangle.' % name)
        self.__ip_tr = var

    @property
    def cp_tr(self):
        return self.__cp_tr

    @cp_tr.setter
    def cp_tr(self, var):
        name = 'cp_tr'
        var = hf.ndarray_try_convert(var, name, logger, type=float)
        hf.check_condition(var.shape[0], var.shape[1], name, logger)
        nans = np.isnan(var)
        j = var.shape[1]
        ix = np.tile(np.arange(0, j), j).reshape(j, j) + np.tile(np.arange(1, j + 1), j).reshape(j, j).T
        if np.sum(nans) > 0:
            assert np.min(ix[nans]) > j, logger.error(
                'Not valid values in %s upper triangle.' % name)
        self.__cp_tr = var

    @property
    def in_tr(self):
        return self.__in_tr

    @in_tr.setter
    def in_tr(self, var):
        name = 'in_tr'
        var = hf.ndarray_try_convert(var, name, logger, type=float)
        hf.check_condition(var.shape[0], var.shape[1], name, logger)
        nans = np.isnan(var)
        j = var.shape[1]
        ix = np.tile(np.arange(0, j), j).reshape(j, j) + np.tile(np.arange(1, j + 1), j).reshape(j, j).T
        if np.sum(nans) > 0:
            assert np.min(ix[nans]) > j, logger.error(
                'Not valid values in %s upper triangle.' % name)
        self.__in_tr = var

    @property
    def cn_tr(self):
        return self.__cn_tr

    @cn_tr.setter
    def cn_tr(self, var):
        name = 'cn_tr'
        var = hf.ndarray_try_convert(var, name, logger, type=float)
        hf.check_condition(var.shape[0], var.shape[1], name, logger)
        nans = np.isnan(var)
        j = var.shape[1]
        ix = np.tile(np.arange(0, j), j).reshape(j, j) + np.tile(np.arange(1, j + 1), j).reshape(j, j).T
        if np.sum(nans) > 0:
            assert np.min(ix[nans]) > j, logger.error(
                'Not valid values in %s upper triangle.' % name)
        self.__cn_tr = var

    @property
    def j(self):
        return self.__j

    @j.setter
    def j(self, var):
        self.__j = hf.triangles_dimension(incremental_payments=self.ip_tr,
                                          cased_payments=self.cp_tr,
                                          incurred_number=self.in_tr,
                                          cased_number=self.cn_tr)

    @property
    def ix(self):
        return self.__ix

    @ix.setter
    def ix(self, var):
        self.__ix = np.tile(np.arange(0, self.j), self.j).reshape(self.j,
                                                                  self.j) + np.tile(np.arange(1, self.j + 1),
                                                                                    self.j).reshape(self.j,
                                                                                                    self.j).T
    @property
    def reported_claims(self):
        return self.__reported_claims

    @reported_claims.setter
    def reported_claims(self, var):
        name = 'reported_claims'
        var = hf.ndarray_try_convert(var, name, logger)
        hf.check_condition(var.shape[0], self.j, name, logger)
        self.__reported_claims = var

class ReservingModel:
    """
    Reserving model assumptions.

    """

    def __init__(self,
                 tail=False,
                 reserving_method="fisher_lange",
                 claims_inflation=None,
                 mixing_fq_par=None,
                 mixing_sev_par=None,
                 czj=None):

        self.tail = tail
        self.reserving_method = reserving_method
        self.claims_inflation = claims_inflation

        self.mixing_fq_par = mixing_fq_par
        self.mixing_sev_par = mixing_sev_par

        self.gamma1 = self._noise_variable_setup(parameters=self.mixing_fq_par)
        self.gamma2 = self._noise_variable_setup(parameters=self.mixing_sev_par)
        self.gamma3 = stats.gamma
        self.pois = stats.poisson

        self.czj = czj

        self.model_class = self._model_class()

    @property
    def tail(self):
        return self.__tail

    @tail.setter
    def tail(self, var):
        hf.assert_type_value(var, 'tail', logger, bool)
        self.__tail = var

    @property
    def reserving_method(self):
        return self.__reserving_method

    @reserving_method.setter
    def reserving_method(self, var):
        hf.assert_member(var, config.RESERVING_METHOD, logger)
        self.__reserving_method = var.lower()

    @property
    def mixing_fq_par(self):
        return self.__mixing_fq_par

    @mixing_fq_par.setter
    def mixing_fq_par(self, var):
        name = 'mixing_fq_par'
        if self.reserving_method == "crm":
            hf.assert_type_value(var, name, logger, dict)
            assert all(item in list(var.keys()) for item in ["scale", "a"]), logger.error(
                "%s must contain 'a' and 'scale' parameters. See %s." % (name, config.SITE_LINK))
        self.__mixing_fq_par = var

    @property
    def mixing_sev_par(self):
        return self.__mixing_sev_par

    @mixing_sev_par.setter
    def mixing_sev_par(self, var):
        name = 'mixing_sev_par'
        if self.reserving_method == "crm":
            hf.assert_type_value(var, name, logger, dict)
            assert all(item in list(var.keys()) for item in ["scale", "a"]), logger.error(
                "%s must contain 'a' and 'scale' parameters. See %s." % (name, config.SITE_LINK))
        self.__mixing_sev_par = var

    def _model_class(self):
        if self.reserving_method in ['fisher_lange', 'crm']:
            return 'average_cost'

    def _noise_variable_setup(self, parameters):
        if self.reserving_method == 'crm':
            return stats.gamma(**parameters)
        else:
            return None

class LossReserve:
    """
    Claims loss reserving. The available reserving models are the deterministic Fisher-Lange and the collective risk model.
    Input company data must be ``numpy.ndarray`` data on numbers and payments must be in triangular form:
    two-dimensional ``numpy.ndarray`` with shape (I, J) where I=J.

    :param tail: set it to True when the tail estimate is required. Default False.
    :type tail: ``bool``
    :param reserving_method: one of the reserving methods supported by the GemAct package.
    :type reserving_method: ``str``
    :param claims_inflation: claims inflation. In case no tail is present and the triangular data IxJ matrices,
                            claims_inflation must be J-1 dimensional. When a tail estimate is required, it must be
                            J dimensional. In case no tail is present it must be J-1 dimensional.
    :type claims_inflation: ``numpy.ndarray``
    :param czj: severity coefficient of variation by development period.
                It is set to None in case the crm is selected as
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
        * *incremental_payments* = (``numpy.ndarray``) --
            Incremental payments' triangle
        * *cased_payments* = (``numpy.ndarray``) --
            Cased payments triangle
        * *incurred_number* = (``numpy.ndarray``) --
            Incurred number
        * *cased_number* = (``numpy.ndarray``) --
            Cased number
        * *reported_claims* = (``numpy.ndarray``) --
            Number of reported claims by accident period.
            Data must be provided from old to recent.

    """

    def __init__(self,
                 data,
                 reservingmodel,
                 custom_alphas=None,
                 custom_ss=None,
                 ntr_sim=1000,
                 set_seed=42):


        self.data = data
        self.reservingmodel = reservingmodel

        self.czj = self.reservingmodel.czj
        self.czj_t = self._triangular_czj()

        self.ap_tr = None

        # variables and other objects
        self.ntr_sim = ntr_sim
        self.random_state = set_seed

        # attributes with opportunities of not standard customization
        self.alpha_fl = custom_alphas if custom_alphas is not None else self._alpha_computer()
        self.ss_fl_ = custom_ss if custom_ss is not None else self._ss_computer()

        self.reserve, self.m_sep, self.skewness = self._lossreserving()

    @property
    def czj(self):
        return self.__czj

    @czj.setter
    def czj(self, var):
        name = 'czj'
        if var is None:
                var = np.repeat(.001, self.data.j + self.reservingmodel.tail)
        else:
            var = hf.ndarray_try_convert(var, name, logger, type=float)
            check = self.data.j if self.reservingmodel.tail else self.data.j - 1
            hf.check_condition(var.shape[0], check, name, logger)
            var = np.concatenate(([.0], var))

        self.__czj = var

    @property
    def ntr_sim(self):
        return self.__ntr_sim

    @ntr_sim.setter
    def ntr_sim(self, var):
        name = 'ntr_sim'
        hf.assert_type_value(var, name, logger, (float, int))
        var = int(var)
        self.__ntr_sim = var

    @property
    def random_state(self):
        return self.__random_state

    @random_state.setter
    def random_state(self, var):
        self.__random_state = hf.handle_random_state(var, logger)

    @property
    def ap_tr(self):
        return self.__ap_tr

    @ap_tr.setter
    def ap_tr(self, var):
        if self.reservingmodel.model_class == 'average_cost':
            temp = self.data.ip_tr / self.data.in_tr
            for i in range(1, self.data.j):
                temp[self.data.j - i:, i] = temp[self.data.j - 1 - i, i] * np.cumprod(self.reservingmodel.claims_inflation[:i])

            if self.reservingmodel.tail:
                var = np.column_stack((temp, temp[0, -1] * np.cumprod(self.reservingmodel.claims_inflation)))
            else:
                var = temp
        self.__ap_tr = var

    # attributes with not-standard customization
    @property
    def alpha_fl(self):
        return self.__alpha_fl

    @alpha_fl.setter
    def alpha_fl(self, var):
        name = 'alpha_fl'
        var = hf.ndarray_try_convert(var, name, logger)
        hf.check_condition(
            var.shape[0],self.data.j + self.reservingmodel.tail - 1, name, logger
        )
        self.__alpha_fl = var

    @property
    def ss_fl_(self):
        return self.__ss_fl_

    @ss_fl_.setter
    def ss_fl_(self, var):
        name = 'ss_fl_'
        var = hf.ndarray_try_convert(var, name, logger)
        hf.check_condition(
            var.shape[0],self.data.j + self.reservingmodel.tail - 1, name, logger
        )
        assert np.abs(np.sum(var) - 1) < 1e+04, logger.error(
            'Make sure the settlement speed vector sums to one.')
        self.__ss_fl_ = var

    # methods
    def _triangular_czj(self):
        #czj below is from the lossaggregation class
        return np.tile(self.czj, self.data.j).reshape(self.data.j, -1)

    def _alpha_computer(self):
        """
        Fisher-Lange alpha. Given a JxJ triangle, this is going to be
        J-1 dimensional in case no tail is present and J dimensional in case of tail estimates.

        :return: vectors of alpha
        :rtype: ``numpy.ndarray``
        """
        temp_in_ = self.data.in_tr.copy()
        temp_cn_ = self.data.cn_tr.copy()
        temp_in_[self.data.ix >self.data.j] = 0.
        temp_cn_[self.data.ix >self.data.j] = np.nan
        v_ = np.flip(np.apply_along_axis(arr=temp_in_, func1d=np.sum, axis=1))
        # create a matrix with the cumulative number of claims in the diagonal
        tot_inc = np.rot90(np.array([v_, ] *self.data.j))
        # create a matrix with cumulative sums
        cum_inc = np.apply_along_axis(arr=temp_in_, func1d=np.cumsum, axis=1)
        # create a matrix
        dg_ = np.flip(np.diag(np.rot90(temp_cn_)))
        mx2_ = np.rot90(np.array([dg_, ] *self.data.j))
        # create the matrix of claims incurred in the future years
        mx1_ = (tot_inc - cum_inc + mx2_) / temp_cn_

        mx1_[self.data.ix >=self.data.j] = np.nan

        if not self.reservingmodel.tail:
            return np.apply_along_axis(arr=mx1_[:, :-1], func1d=np.nanmean, axis=0)
        else:
            return np.concatenate((np.apply_along_axis(arr=mx1_[:, :-1], func1d=np.nanmean, axis=0), [1.]))

    def _ss_computer(self):
        """
        Fisher-Lange settlement speeds. Given a JxJ triangle, this is going to be
        J-1 dimensional in case no tail is present and J dimensional in case of tail estimates.

        :return: settlement speed
        :rtype: ``numpy.ndarray``
        """
        temp = np.flip((np.diag(np.rot90(self.data.in_tr)) * self.data.reported_claims[-1] / self.data.reported_claims)[:-1])

        if self.reservingmodel.tail:
            temp = np.concatenate((temp, [self.data.cn_tr[0, -1] * self.data.reported_claims[-1] / self.data.reported_claims[0]]))

        return temp / np.sum(temp)

    def _ss_triangle(self):
        """
        Fisher-Lange settlement speed array into a triangle. Given a JxJ triangle, this is going to be
        JxJ-1 dimensional in case no tail is present and JxJ dimensional in case of tail estimates.

        :return: settlement speed triangle
        :rtype: ``numpy.ndarray``
        """
        mx1_ = np.array([np.concatenate(([0.], self.ss_fl_)), ] *self.data.j)
        if self.reservingmodel.tail:
            new_ix = np.concatenate((self.data.ix <=self.data.j, np.repeat(False, self.data.j).reshape(self.data.j, 1)), axis=1)
            mx1_[new_ix] = np.nan
        else:
            mx1_[self.data.ix <=self.data.j] = np.nan

        return np.apply_along_axis(arr=mx1_, func1d=hf.normalizernans, axis=1)

    def _fill_numbers(self):
        """
        Lower triangle of numbers. Given a JxJ triangle, this is going to be
        JxJ dimensional in case no tail is present and Jx(J+1) dimensional in case of tail estimates.

        :return: number of payments
        :rtype: ``numpy.ndarray``
        """
        # quotas
        if not self.reservingmodel.tail:
            v_ = np.concatenate((self.alpha_fl, [1.]))
        else:
            v_ = self.alpha_fl
        alq_ = np.flip(np.array([v_, ] *self.data.j).T)

        # diagonal
        dg_ = np.diag(np.flip(np.rot90(self.data.cn_tr)))
        amounts_ = np.flip(np.array([dg_, ] *self.data.j).T)

        if self.reservingmodel.tail:
            alq_ = np.column_stack((np.ones(self.data.j), alq_))
            amounts_ = np.column_stack((np.ones(self.data.j), amounts_))

        # final development
        ss_amounts_alq = self.ss_tr * amounts_ * alq_

        if self.reservingmodel.tail:
            new_ix = np.concatenate((self.data.ix >self.data.j, np.repeat(True,self.data.j).reshape(self.data.j, 1)), axis=1)
            temp_ = np.column_stack((self.data.in_tr, np.ones(self.data.j)))
            temp_[new_ix] = ss_amounts_alq[~np.isnan(ss_amounts_alq)]
            return temp_
        else:
            temp_ = self.data.in_tr.copy()
            temp_[self.data.ix >self.data.j] = ss_amounts_alq[~np.isnan(ss_amounts_alq)]
            return temp_

    def _fisherlange(self):
        """
        Loss reserve computed with the fisher-lange reserving model.

        :return: fisher-lange reserve
        :rtype: ``numpy.float64``
        """

        self.predicted_i_payments = self.predicted_i_numbers * self.ap_tr

        if self.reservingmodel.tail:
            new_ix = np.concatenate((self.data.ix >self.data.j, np.repeat(True,self.data.j).reshape(self.data.j, 1)), axis=1)
            self.predicted_i_payments[0, -1] = self.data.cp_tr[0, -1]
            return np.sum(self.predicted_i_payments[new_ix]), None, None
        else:
            return np.sum(self.predicted_i_payments[self.data.ix >self.data.j]), None, None

    def _stochastic_crm(self):
        """
        Loss reserve computed with the collective risk model based on the fisher-lange.

        :return: reserve prediction (simulations mean), reserve m_sep prediction, reserve skewness
        :rtype:``numpy.float64``,``numpy.float64``,``numpy.float64``

        """

        flag_ = np.repeat('ay' + str(0), self.ap_tr.shape[1])  # create a flag that will be used to pick the correct ay
        for ay in range(1, self.ap_tr.shape[0]):
            cell_ = 'ay' + str(ay)
            temp_ = np.repeat(cell_, self.ap_tr.shape[1])
            flag_ = np.vstack((flag_, temp_))

        if not self.reservingmodel.tail:

            v1_ = self.predicted_i_numbers[self.data.ix >self.data.j]  # numbers lower triangle
            v2_ = self.ap_tr[self.data.ix >self.data.j]  # average payments lower triangle
            czj_v = self.czj_t[self.data.ix > self.data.j]  # coefficient of variation lower triangle
            flag_v = flag_[self.data.ix >self.data.j]
        else:
            new_ix = np.concatenate((self.data.ix >self.data.j, np.repeat(True, self.data.j).reshape(self.data.j, 1)),
                                    axis=1)  # lower triangle and tail
            v1_ = self.predicted_i_numbers[new_ix]  # numbers lower triangle and tail
            v2_ = self.ap_tr[new_ix]  # average payments lower triangle and tail
            czj_v = self.czj_t[new_ix]  # coefficient of variation lower triangle and tail
            flag_v = flag_[new_ix]

        np.random.seed(self.random_state)
        output = np.array([])
        # now = datetime.now()
        self.crm_sep_ay = np.array([], dtype=np.float64)  # to store the mean squared error of prediction
        self.crm_ul_ay = np.array([], dtype=np.float64)  # to store the ultimate cost
        self.ay_reserve_crm = np.array([], dtype=np.float64)  # to store the reserve by time period
        mean_squared_ep_temp_ = np.array([], dtype=np.float64)
        ultimate_temp = np.array([], dtype=np.float64)
        for i in range(0, len(v1_)):  # crm computed on each lower triangle cell

            f_ = flag_v[i]  # flag the cell with the correspondent ay

            if i + 1 < len(flag_v):
                fp_ = flag_v[
                    i + 1]
            else:
                fp_ = 'stop'

            p1_ = v1_[i]  # cell numbers
            p2_ = v2_[i]  # cell average payment
            p3_ = czj_v[i]  # cell coefficient of variation

            vec1_ = p1_ * self.reservingmodel.gamma1.rvs(self.ntr_sim)
            vec2_ = p2_ ** 2 / (p3_ * self.reservingmodel.gamma2.rvs(self.ntr_sim))
            vec3_ = p3_ * self.reservingmodel.gamma2.rvs(self.ntr_sim) / p2_

            vec4_ = np.apply_along_axis(func1d=hf.lrcrm_f1, arr=vec1_.reshape(-1, 1), axis=1, dist=self.reservingmodel.pois).reshape(
                -1, )  # simulate all the CRMs for the cell
            mx_ = np.array([vec4_, vec2_, vec3_]).T  # create a matrix of parameters
            temp_ = np.apply_along_axis(axis=1, arr=mx_, func1d=hf.lrcrm_f2,
                                        dist=self.reservingmodel.gamma3)  # simulate the reserves
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

        print("")

        if not self.reservingmodel.tail:
            self.ay_reserve_crm = np.concatenate(([0], self.ay_reserve_crm))
            self.crm_sep_ay = np.concatenate(([0], self.crm_sep_ay))
            self.crm_ul_ay = np.concatenate(([0], self.crm_ul_ay))
        else:  # I need to fill the last line of the ultimate in advance in case of CRM
            diagonal_cml_ = self.predicted_i_payments[-1, 0]
            self.crm_ul_ay[self.predicted_i_payments.shape[0] - 1] = self.crm_ul_ay[
                                                                         self.predicted_i_payments.shape[
                                                                             0] - 1] + diagonal_cml_

        for ay in range(0, self.predicted_i_payments.shape[
                               0] - self.reservingmodel.tail):
            # add to the estimated random cumulative payments the upper triangle amounts
            diagonal_cml_ = np.cumsum(self.predicted_i_payments[ay, :(self.data.j - ay - 1 - self.reservingmodel.tail + 1)])[
                -1]  # separate it to make it readable
            self.crm_ul_ay[ay] = self.crm_ul_ay[ay] + diagonal_cml_

        reserves_ = np.apply_along_axis(arr=output, func1d=np.sum, axis=1)
        return np.mean(reserves_), np.std(reserves_), stats.skew(reserves_)

    def _lossreserving(self):

        if self.reservingmodel.model_class == 'average_cost':
            self.ss_tr = self._ss_triangle()  # triangle of settlement speed
            self.predicted_i_numbers = self._fill_numbers()  # individual payments triangle

            if self.reservingmodel.reserving_method == 'fisher_lange':
                return self._fisherlange()

            elif self.reservingmodel.reserving_method == 'crm':
                self.fl_reserve, _, _ = self._fisherlange()
                return self._stochastic_crm()

    def ss_plot(self, start_=0):
        """
        Plot the settlement speed vector for each accident period.

        :param start_: starting accident period from which to plot.
        :type start_: ``int``
        """

        x_ = np.arange(0,self.data.j + self.reservingmodel.tail)
        plt.title('Plot of the settlement speed from accident period %s' % start_)
        plt.xlabel('Development period')
        plt.ylabel('Settlement speed')
        for i in range(start_, self.data.j):
            plt.plot(x_, self.ss_tr[i, :], '-.', label='AP %s' % i)
            plt.legend()
        plt.show()


    def alpha_plot(self):
        """
        Plot the Fisher-Lange alpha.
        """
        x_ = np.arange(0,self.data.j + self.reservingmodel.tail - 1)
        plt.title('Plot of Alpha')
        plt.plot(x_, self.alpha_fl, '-.', label='Alpha')
        plt.xlabel('Development period')
        plt.ylabel('Alpha')
        plt.show()

    def _reserve_by_ay_fl(self):
        """
        The fisher-lange reserve computed for each accident period and
        the fisher-lange ultimate cost for each accident period.

        :return: reserve for each accident period,ultimate cost for each accident period
        :rtype: ``numpy.ndarray``, ``numpy.ndarray``
        """
        self.ay_reserveFL = np.array([])
        self.ay_ultimateFL = np.array([])
        for ay in range(0, self.predicted_i_payments.shape[0]):
            v_ = self.predicted_i_payments[ay, :]
            self.ay_reserveFL = np.concatenate((self.ay_reserveFL, [np.sum(v_[(self.data.j - ay):])]))
            self.ay_ultimateFL = np.concatenate((self.ay_ultimateFL, [np.cumsum(v_)[-1]]))

    def _build_base_print(self):

        if self.reservingmodel.model_class == 'average_cost':
            self._reserve_by_ay_fl()
            ay_ = np.arange(0, self.predicted_i_payments.shape[0])
            ultimate = np.round(self.ay_ultimateFL, 2)
            reserve = np.round(self.ay_reserveFL, 2)

        return np.dstack((ay_, ultimate, reserve)).reshape(-1, 3), ['time', 'ultimate FL', 'reserve FL']

    def _build_graphic_parameters(self):
        if self.reservingmodel.model_class == 'average_cost':
            s_ = "{: >20} {: >20} {: >20} {: >20}"

        return s_


    def _build_comparison_print(self, data, l_, s_):

        if self.reservingmodel.reserving_method=='crm':
            #build data print
            data2 = np.dstack((np.round(self.crm_ul_ay, 2), np.round(self.ay_reserve_crm, 2), np.round(
                self.crm_sep_ay, 2))).reshape(-1, 3)
            data = np.column_stack((data, data2))

            #build header
            l2_ = ['ultimate CRM', 'reserve CRM', 'm_sep CRM']
            l_.extend(l2_)
            s_ = s_ + " {: >20} {: >20} {: >20}"

        return data, l_, s_

    def _print_total_reserve(self):

        if self.reservingmodel.model_class == 'average_cost':
            print('\n FL reserve: ', np.round(self.fl_reserve, 2))
            if self.reservingmodel.reserving_method == 'crm':
                print('\n CRM reserve: ', np.round(self.reserve, 2))
                print('\n CRM m_sep: ', np.round(self.m_sep, 2))


    def print_loss_reserve_specs(self):
        """
        Table with claims reserve results.
        When the stochastic reserve according to the collective risk model is computed the results
        are compared with the Fisher-Lange.

        """
        data, l_ = self._build_base_print()
        s_ = self._build_graphic_parameters()
        data, l_, s_=self._build_comparison_print(data, l_, s_)

        print(s_.format(" ", *l_))
        print("{: >20} {: >20}".format(" ", *[
            " ================================================================"
            "===================================================================="]))
        for row in data:
            print(s_.format("", *row))

        self._print_total_reserve()

        return


    def mean(self):
        """
        Mean of the loss reserve.
        Depending on the selected reserving method, it returns either the attribute crm_reserve or fl_reserve.

        :return: mean of the loss reserve.
        :rtype: ``numpy.float64``
        """
        return self.reserve

    def std(self):
        """
        Standard deviation of the loss reserve (not available for claims reserving with the fisher lange).

        :return: standard deviation of the loss reserve.
        :rtype: ``numpy.float64``
        """

        return self.m_sep

    def skewness(self):
        """
        Skewness of the loss reserve (not available for claims reserving with the fisher lange).

        :return: skewness of the loss loss.
        :rtype: ``numpy.float64``
        """
        return self.skewness
