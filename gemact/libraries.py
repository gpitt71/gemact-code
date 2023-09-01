import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy import stats
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.linalg import cholesky
from scipy.optimize import root_scalar
import sys
import time
from twiggy import quick_setup, log
from itertools import groupby
