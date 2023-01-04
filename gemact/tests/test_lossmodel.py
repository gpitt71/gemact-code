# import unittest
# import gemact
#
# class TestLossModel(unittest.TestCase):
#     def test_computational_methods(self):
#         decimalPlace = 3
#
#         frequency = gemact.Frequency(
#             dist='poisson',
#             par={'mu': 4}
#         )
#
#         # define a Generalized Pareto severity model
#         severity = gemact.Severity(
#             dist='genpareto',
#             par={'c': .2, 'scale': 1}
#         )
#
#         policystructure = gemact.PolicyStructure()
#
#         lossmodel_dft = gemact.LossModel(
#             frequency=frequency,
#             severity=severity,
#             policystructure=policystructure,
#             aggr_loss_dist_method='fft',
#             n_sev_discr_nodes=int(1000),
#             sev_discr_step=1,
#             n_aggr_dist_nodes=int(10000)
#         )
#
#
#         lossmodel_rec = gemact.LossModel(
#             frequency=frequency,
#             severity=severity,
#             policystructure=policystructure,
#             aggr_loss_dist_method='recursion',
#             n_sev_discr_nodes=int(1000),
#             sev_discr_step=1,
#             n_aggr_dist_nodes=int(10000)
#         )
#
#         lossmodel_mc = gemact.LossModel(
#             frequency=frequency,
#             severity=severity,
#             policystructure=policystructure,
#             aggr_loss_dist_method='mc',
#             n_sim=1e+05,
#             random_state=1
#         )
#
#         self.assertAlmostEqual(lossmodel_rec.mean(), lossmodel_dft.mean(), decimalPlace)
#         self.assertAlmostEqual(lossmodel_mc.mean(), lossmodel_dft.mean(), decimalPlace)
#
#
