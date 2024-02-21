import unittest
from gemact import lossmodel
import numpy as np

class TestLossModel(unittest.TestCase):
    def test_computational_methods(self):
        decimalPlace = 2

        frequency = lossmodel.Frequency(
            dist='poisson',
            par={'mu': 4}
        )

        # define a Generalized Pareto severity model
        severity = lossmodel.Severity(
            dist='genpareto',
            par={'c': .2, 'scale': 1}
        )

        policystructure = lossmodel.PolicyStructure()

        lossmodel_dft = lossmodel.LossModel(
            frequency=frequency,
            severity=severity,
            policystructure=policystructure,
            aggr_loss_dist_method='fft',
            n_sev_discr_nodes=int(80),
            sev_discr_step=1,
            n_aggr_dist_nodes=int(256)
        )




        lossmodel_rec = lossmodel.LossModel(
            frequency=frequency,
            severity=severity,
            policystructure=policystructure,
            aggr_loss_dist_method='recursion',
            n_sev_discr_nodes=int(80),
            sev_discr_step=1,
            n_aggr_dist_nodes=int(256)
        )



        lossmodel_mc = lossmodel.LossModel(
            frequency=frequency,
            severity=severity,
            policystructure=policystructure,
            aggr_loss_dist_method='mc',
            n_sim=1e+05,
            random_state=1
        )



        self.assertAlmostEqual(lossmodel_rec.mean(), lossmodel_dft.mean(), decimalPlace)
        self.assertAlmostEqual(lossmodel_mc.mean(), lossmodel_dft.mean(), decimalPlace)

    def test_severity_discretization(self):

        severity = lossmodel.Severity(
            dist='gamma',
            par={'a': 5}
        )

        ld=severity.discretize(
            discr_method='lowerdiscretization',
            n_discr_nodes=50,
            discr_step= 1)

        ud = severity.discretize(
            discr_method='upperdiscretization',
            n_discr_nodes=50,
            discr_step= 1)

        md = severity.discretize(
            discr_method='massdispersal',
            n_discr_nodes=50,
            discr_step= 1)

        lm = severity.discretize(
            discr_method='localmoments',
            n_discr_nodes=50,
            discr_step= 1)

        print('Sum of probabilities md', np.sum(md['fj']))
        print('probabilities ld', md['fj'][:5])
        print('shape md', md['fj'].shape)

        print('Sum of probabilities ld' ,np.sum(ld['fj']))
        print('probabilities ld', ld['fj'][:5])
        print('shape ld', ld['fj'].shape)

        print('Sum of probabilities ud', np.sum(ud['fj']))
        print('probabilities ud', ud['fj'][:5])
        print('shape ud', ud['fj'].shape)

        print('Sum of probabilities ud', np.sum(lm['fj']))
        print('probabilities ud', lm['fj'][:5])
        print('shape ud', lm['fj'].shape)


