import unittest
import numpy as np
from gemact import gemdata
from gemact import distributions
from gemact.lossmodel import Frequency, Severity, LossModel, PolicyStructure, Layer
from gemact.lossreserve import AggregateData, ReservingModel, LossReserve
from gemact.lossaggregation import LossAggregation
from gemact import copulas


class TestLossModel(unittest.TestCase):

    def test_severity_discretization(self):

        severity = Severity(
            dist='gamma',
            par={'a': 5}
        )

        massdispersal = severity.discretize(
            discr_method='massdispersal',
            n_discr_nodes=5000,
            discr_step=.01,
            deductible=0,
            cover=100
        )

        discrete_mean = np.sum(massdispersal['nodes'] * massdispersal['fj'])
        decimalPlace = 3
        self.assertAlmostEqual(discrete_mean, 4.999999999999998, decimalPlace)

        severity.plot_discr_sev_cdf(
            discr_method = 'massdispersal',
            n_discr_nodes = 50,
            discr_step = 1,
            deductible = 0,
            cover = 40
            )


    def test_distributions(self):

        ztpois = distributions.ZTPoisson(mu=2)

        random_variates = ztpois.rvs(int(1e+05), random_state=1)
        decimalPlace = 3
        self.assertAlmostEqual(np.mean(random_variates), 2.3095, decimalPlace)

        clayton_copula = copulas.ClaytonCopula(par=1.2, dim=2)
        values = np.array([[.5, .5]])
        self.assertAlmostEqual(clayton_copula.cdf(values)[0], 0.3443010799531017, decimalPlace)

    def test_lossmodel(self):

        frequency = Frequency(
            dist='poisson',
            par={'mu': 4}
        )

        severity = Severity(
            dist='gamma',
            par={'a': 5}
        )

        lm_mc = LossModel(frequency = frequency,
                          severity = severity,
                          aggr_loss_dist_method = 'mc',
                          n_sim = 1e+04,
                          random_state = 1
                          )

        lm_mc.plot_dist_cdf(color='orange')
        decimalPlace = 3
        self.assertAlmostEqual(lm_mc.moment(central=False, n=1), 19.950968442875176, decimalPlace)
        self.assertAlmostEqual(lm_mc.mean(), 19.950968442875176, decimalPlace)
        self.assertAlmostEqual(lm_mc.ppf(q=[.8, .7])[0], 28.83665586, decimalPlace)

        policystructure = PolicyStructure(
            layers=Layer(
                cover=50,
                deductible=2)
        )

        lm_XL = LossModel(
            frequency=frequency,
            severity=severity,
            policystructure=policystructure,
            aggr_loss_dist_method='fft',
            sev_discr_method='massdispersal',
            n_sev_discr_nodes=int(10000),
            sev_discr_step=.01,
            n_aggr_dist_nodes=int(100000)
        )
        self.assertAlmostEqual(lm_XL.pure_premium[0], 12.08995159320356, decimalPlace)

        policystructure_RS = PolicyStructure(
            layers=Layer(
                cover=100,
                deductible=0,
                aggr_deductible=100,
                reinst_loading=1,
                n_reinst=2
            ))

        lm_RS = LossModel(
            frequency=Frequency(
                dist='poisson',
                par={'mu': .5}
            ),
            severity=Severity(
                par={'loc': 0,
                     'scale': 83.34,
                     'c': 0.834},
                dist='genpareto'
            ),
            policystructure=policystructure_RS,
            aggr_loss_dist_method='fft',
            sev_discr_method='massdispersal',
            n_sev_discr_nodes=int(10000),
            sev_discr_step=.01,
            n_aggr_dist_nodes=int(100000)
        )

        pure_premium = lm_RS.pure_premium[0]

        self.assertAlmostEqual(pure_premium, 4.318987283037852, decimalPlace)


    def test_lr_codechunk1(self):

        ip_ = gemdata.incremental_payments
        in_ = gemdata.incurred_number
        cp_ = gemdata.cased_payments
        cn_ = gemdata.cased_number
        reported_ = gemdata.reported_claims
        claims_inflation = gemdata.claims_inflation

        ad = AggregateData(
        incremental_payments = ip_,
        cased_payments = cp_,
        cased_number = cn_,
        reported_claims = reported_,
        incurred_number = in_)

        return ad, claims_inflation

    def test_lr_codechunk2(self):

        _, claims_inflation = self.test_lr_codechunk1()

        rm = ReservingModel(tail=True,
        reserving_method = "fisher_lange",
        claims_inflation = claims_inflation)

        return rm


    def test_lr_codechunk_fl(self):
        ad, _ = self.test_lr_codechunk1()
        rm = self.test_lr_codechunk2()

        lr = LossReserve(data=ad,
                         reservingmodel = rm)

        decimalPlace = 2
        self.assertAlmostEqual(lr.mean(), 254973.71, decimalPlace)

        # lr.print_loss_reserve()
        # lr.plot_alpha_fl()
        # lr.plot_ss_fl(start_=7)


    def test_lr_codechunk3(self):
        _, claims_inflation = self.test_lr_codechunk1()

        mixing_fq_par = {'a': 1 / .08 ** 2,  # mix frequency
                         'scale': .08 ** 2}

        mixing_sev_par = {'a': 1 / .08 ** 2, 'scale': .08 ** 2}  # mix severity
        czj = gemdata.czj
        claims_inflation = gemdata.claims_inflation

        rm = ReservingModel(tail=True,
                           reserving_method="crm",
                           claims_inflation=claims_inflation,
                           mixing_fq_par=mixing_fq_par,
                           mixing_sev_par=mixing_sev_par,
                           czj=czj)

        return rm

    def test_lr_codechunk_crm(self):
        ad, _ = self.test_lr_codechunk1()
        rm = self.test_lr_codechunk3()

        lr = LossReserve(data=ad,
        reservingmodel = rm,
        ntr_sim = 100,
        random_state=1)


        decimalPlace=3
        self.assertAlmostEqual(lr.mean(), 257340.0609855847, decimalPlace)


        lr.std()
        lr.skewness()

    def test_la(self):

        la = LossAggregation(
        margins = ['genpareto', 'genpareto'],
        margins_pars = [{'loc': 0, 'scale':1 / .8, 'c': 1 / .8},
        {'loc': 0,
         'scale': 1 / 2,
          'c': 1 / 2}
        ],
        copula = 'clayton',
        copula_par = {'par': .4,
                      'dim': 2})


        decimalPlace=3

        self.assertAlmostEqual(la.cdf(1,method='aep',n_iter=8), 0.26972015, decimalPlace)

    def test_ClaytonCopula(self):
        clayton_copula = copulas.ClaytonCopula(par=1.2, dim=2)
        values = np.array([[.5, .5]])

        print('Clayton copula cdf ', clayton_copula.cdf(values)[0])
