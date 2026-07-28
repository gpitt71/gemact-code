import unittest
import gemact.lossreserve


class TestLossModel(unittest.TestCase):

    def _make_aggregate_data(self):
        return gemact.AggregateData(
            incremental_payments=gemact.gemdata.incremental_payments,
            cased_payments=gemact.gemdata.cased_payments,
            open_claims_number=gemact.gemdata.open_number,
            reported_claims=gemact.gemdata.reported_claims,
            payments_number=gemact.gemdata.payments_number,
        )

    def _make_reserving_model_fl(self):
        return gemact.ReservingModel(
            tail=True,
            reserving_method="fisher_lange",
            claims_inflation=gemact.gemdata.claims_inflation,
        )

    def _make_reserving_model_crm(self):
        mixing_fq_par = {
            "a": 1 / 0.08**2,
            "scale": 0.08**2,
        }

        mixing_sev_par = {
            "a": 1 / 0.08**2,
            "scale": 0.08**2,
        }

        return gemact.ReservingModel(
            tail=True,
            reserving_method="crm",
            claims_inflation=gemact.gemdata.claims_inflation,
            mixing_fq_par=mixing_fq_par,
            mixing_sev_par=mixing_sev_par,
            czj=gemact.gemdata.czj,
        )

    def test_AggregateData(self):
        ad = self._make_aggregate_data()

        self.assertIsInstance(ad, gemact.AggregateData)

    def test_ReservingModelFL(self):
        rm = self._make_reserving_model_fl()

        self.assertIsInstance(rm, gemact.ReservingModel)
        self.assertEqual(rm.reserving_method, "fisher_lange")

    def test_ReservingModelCRM(self):
        rm = self._make_reserving_model_crm()

        self.assertIsInstance(rm, gemact.ReservingModel)
        self.assertEqual(rm.reserving_method, "crm")

    def test_fisherlange(self):
        ad = self._make_aggregate_data()
        rm = self._make_reserving_model_fl()

        lr = gemact.LossReserve(
            data=ad,
            reservingmodel=rm,
        )

        self.assertIsInstance(lr, gemact.LossReserve)

    def test_crm(self):
        ad = self._make_aggregate_data()
        rm = self._make_reserving_model_crm()

        lr = gemact.LossReserve(
            data=ad,
            reservingmodel=rm,
            ntr_sim=2,
        )

        self.assertIsInstance(lr, gemact.LossReserve)