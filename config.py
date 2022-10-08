from .libraries import *
from . import copulas
from . import distributions

SITE_LINK = 'https://gem-analytics.github.io/gemact/'  # imported in lossmodel, lossaggregation
DCEILING = 5  # imported in lossaggregation
LOSS_AGGREGATION_METHOD_LIST = ['aep', 'mc']  # imported in lossaggregation
SEV_DISCRETIZATION_METHOD_LIST = ['massdispersal', 'localmoments']  # imported in lossmodel
AGGREGATE_LOSS_APPROX_METHOD_LIST = ['fft', 'recursion', 'mc']  # imported in lossmodel

COP_DICT = {
    'clayton': copulas.ClaytonCopula,
    'frank': copulas.FrankCopula,
    'gumbel': copulas.GumbelCopula,
    'gaussian': copulas.GaussCopula,
    'tstudent': copulas.TCopula,
    'independent': copulas.IndependentCopula,
    'frechet–hoeffding-lower': copulas.FHLowerCopula,
    'frechet–hoeffding-upper': copulas.FHUpperCopula
    }

DIST_DICT = {
    'binom': distributions.Binom,
    'geom': distributions.Geom,
    'nbinom': distributions.NegBinom,
    'poisson': distributions.Poisson,
    'zmbinom': distributions.ZMBinom,
    'zmgeom': distributions.ZMGeom,
    'zmlogser': distributions.ZMLogser,
    'zmnbinom': distributions.ZMNegBinom,
    'zmpoisson': distributions.ZMPoisson,
    'ztgeom': distributions.ZTGeom,
    'ztbinom': distributions.ZTBinom,
    'ztnbinom': distributions.ZTNegBinom,
    'ztpoisson': distributions.ZTPoisson,
    'beta': distributions.Beta,
    'burr12': distributions.Burr12,
    'dagum': distributions.Dagum,
    'exponential': distributions.Exponential,
    'fisk': distributions.Fisk,
    'gamma': distributions.Gamma,
    'genbeta': distributions.GenBeta,
    'genpareto': distributions.GenPareto,  
    'invgamma': distributions.InvGamma,
    'invgauss': distributions.InvGauss,
    'invweibull': distributions.InvWeibull,
    'lognormal': distributions.Lognormal,
    'weibull': distributions.Weibull
}
