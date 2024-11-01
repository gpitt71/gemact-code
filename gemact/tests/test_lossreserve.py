import unittest
import gemact.lossreserve

class TestLossModel(unittest.TestCase):
    def test_AggregateData(self):

        ip_ = gemact.gemdata.incremental_payments
        in_ = gemact.gemdata.payments_number
        cp_ = gemact.gemdata.cased_payments
        cn_ = gemact.gemdata.open_number
        reported_ = gemact.gemdata.reported_claims

        ad = gemact.AggregateData(
                    incremental_payments=ip_,
                    cased_payments=cp_,
                    open_claims_number =cn_,
                    reported_claims=reported_,
                    payments_number=in_)

        return ad

    def test_ReservingModelFL(self):

        claims_inflation= gemact.gemdata.claims_inflation

        rm = gemact.ReservingModel(tail=True,
                 reserving_method="fisher_lange",
                 claims_inflation=claims_inflation)

        return rm

    def test_ReservingModelCRM(self):

        mixing_fq_par = {'a': 1 / .08 ** 2,  # mix frequency
                         'scale': .08 ** 2}

        mixing_sev_par = {'a': 1 / .08 ** 2, 'scale': .08 ** 2}  # mix severity
        czj = gemact.gemdata.czj
        claims_inflation=gemact.gemdata.claims_inflation

        rm = gemact.ReservingModel(tail=True,
                 reserving_method="crm",
                 claims_inflation=claims_inflation,
                 mixing_fq_par=mixing_fq_par,
                 mixing_sev_par=mixing_sev_par,
                 czj=czj)

        return rm

    def test_fisherlange(self):

        ad = self.test_AggregateData()
        rm = self.test_ReservingModelFL()

        lr = gemact.LossReserve(data=ad,
                                reservingmodel=rm)


    def test_crm(self):
        ad = self.test_AggregateData()
        rm = self.test_ReservingModelCRM()

        lr = gemact.LossReserve(data=ad,
                                reservingmodel=rm,
                                ntr_sim=2)





