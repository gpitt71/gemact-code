from .libraries import *

SITE_LINK = 'https://gem-analytics.github.io/gemact/'
DCEILING = 5  # imported in lossaggregation
LOSS_AGGREGATION_METHOD = {'aep', 'mc'}  # imported in lossaggregation
SEV_DISCRETIZATION_METHOD = {'localmoments', 'massdispersal'}  # imported in lossmodel
AGGREGATE_LOSS_APPROX_METHOD = {'fft', 'mc', 'recursion'}  # imported in lossmodel
POLICY_LAYER_BASIS = {'regular', 'drop-down', 'stretch-down'} # imported in lossmodel
POLICY_LAYER_CATEGORY = {'xlrs', 'xl/sl'} # imported in lossmodel
RESERVING_METHOD = {'fisher_lange', 'crm'} # imported in lossreserve
# dictionary of copulas
COP_DICT = {
    'clayton': 'copulas.ClaytonCopula',
    'frank': 'copulas.FrankCopula',
    'gumbel': 'copulas.GumbelCopula',
    'gaussian': 'copulas.GaussCopula',
    'tstudent': 'copulas.TCopula',
    'independent': 'copulas.IndependenceCopula',
    'frechet–hoeffding-lower': 'copulas.FHLowerCopula',
    'frechet–hoeffding-upper': 'copulas.FHUpperCopula'
    }
# dictionary of distributionss
DIST_DICT = {
    'binom': 'distributions.Binom',
    'geom': 'distributions.Geom',
    'nbinom': 'distributions.NegBinom',
    'poisson': 'distributions.Poisson',
    'zmbinom': 'distributions.ZMBinom',
    'zmgeom': 'distributions.ZMGeom',
    'zmlogser': 'distributions.ZMLogser',
    'zmnbinom': 'distributions.ZMNegBinom',
    'zmpoisson': 'distributions.ZMPoisson',
    'ztgeom': 'distributions.ZTGeom',
    'ztbinom': 'distributions.ZTBinom',
    'ztnbinom': 'distributions.ZTNegBinom',
    'ztpoisson': 'distributions.ZTPoisson',
    'beta': 'distributions.Beta',
    'burr12': 'distributions.Burr12',
    'dagum': 'distributions.Dagum',
    'exponential': 'distributions.Exponential',
    'fisk': 'distributions.Fisk',
    'gamma': 'distributions.Gamma',
    'genbeta': 'distributions.GenBeta',
    'genpareto': 'distributions.GenPareto',  
    'invgamma': 'distributions.InvGamma',
    'invgauss': 'distributions.InvGauss',
    'invweibull': 'distributions.InvWeibull',
    'lognormal': 'distributions.Lognormal',
    'weibull': 'distributions.Weibull'
}
