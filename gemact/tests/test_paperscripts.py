import unittest
from gemact import gemdata
from gemact.lossreserve import AggregateData, ReservingModel, LossReserve
from gemact.lossaggregation import LossAggregation

class TestLossModel(unittest.TestCase):
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

        lr.print_loss_reserve()
        lr.alpha_plot()
        lr.ss_plot(start_=7)


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
        lr.print_loss_reserve()
        lr.mean()
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









