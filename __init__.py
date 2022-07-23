__version__ = "0.1.3"
__doc__ = """GEMAct is an **actuarial package**, 
based on the collective risk theory framework, 
that offers actuarial scientists and practitioners a comprehensive set of tools for 
**non-life** (re)insurance **pricing**, stochastic **claims reserving**, 
**risk aggregation** and extends the set of probability distributions already available in Python. \n
The broad and flexible GEMAct apparatus fits into the expanding community of **Python** programming language. \n
Please visit our [website](https://gem-analytics.github.io/gemact/) to see our documentation and tutorial."""

import gemact.helperfunctions as hf
import gemact.distributions as distributions
import gemact.copulas as copulas
import gemact.gemdata as gemdata
from gemact.config import *
from gemact.lossmodel import *
from gemact.lossreserve import *
from gemact.lossaggregation import *
