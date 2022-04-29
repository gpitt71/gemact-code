__version__= "0.1.2"
__doc__ = "GemAct is an actuarial package, based on the collective risk theory framework, that offers a comprehensive set of tools for non-life (re)insurance pricing, stochastic claims reserving, and risk aggregation.\nThe variety of available functionalities makes GemAct modeling very flexible and provides actuarial scientists and practitioners with a powerful tool that fits into the expanding community of Python programming language."

import gemact.helperfunctions as hfns
import gemact.distributions as distributions
import gemact.gemdata as gemdata
from gemact.lossmodel import *
from gemact.lossreserve import *
from gemact.lossaggregation import *
import gemact.sobol as sbl



