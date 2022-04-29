"""
This script contains helper function to be used in the main scripts "lossmodel.py","lossreserve.py" and "lossaggregation.py".
"""

import numpy as np
from scipy.interpolate import interp1d
from . import sobol as sbl

from twiggy import quick_setup,log
quick_setup()
logger = log.name('hfns')


def ecdf(x_):
    """
    It computes the empirical cumulative density function.

    Empirical cumulative density function computed on the vector x_.

    Parameters:
    x_ (numpy.ndarray): sequence of values to compute the ecdf on.

    Returns:
    x_(numpy.ndarray): starting sequence.
    f(x_)(numpy.ndarray): empirical cumulative density function.
    """
    dim = len(x_)
    x_ = np.sort(x_)
    y_ = np.cumsum(np.repeat(1, dim)) / dim
    f = interp1d(x_, y_)

    return x_, f(x_)

def sobol_generator(n, dim,skip=0):
    """
    Wrapper to generate sobol sequence.

    It generates a dim-dimensional sobol sequence from the script sobol.py.

    Parameters:
    n (int): length of the sobol sequence.
    dim (int): dimension of the sobol sequence.

    Returns:
    numpy.ndarray: generated sobol sequence.

    """
    return sbl.i4_sobol_generate(m=dim, n=n,skip=skip)

#LossReserve

def normalizerNans(x_):
    """
    It normalizes a vector with nan values.

    It normalizes a vector with nan values ignoring the nan values during the computation.

    Parameters:
    x_ (numpy.ndarray): sequence to be normalized.

    Returns:
    numpy.ndarray: normalized sequence.
    """
    if np.sum(np.isnan(x_)) < x_.shape[0]:
        x_[~np.isnan(x_)]=x_[~np.isnan(x_)]/np.sum(x_[~np.isnan(x_)])
    return x_

#LossAggregation

# def iMat(d):
#     if d == 1:
#         return np.array([0, 1])
#     else:
#         temp = iMat(d - 1)
#         if d - 1 == 1:
#             ncol = temp.shape[0]
#         else:
#             ncol = temp.shape[1]
#         return np.concatenate((np.repeat([0, 1], ncol).reshape(1, 2 * ncol), np.tile(temp, 2).reshape(d - 1, 2 * ncol)),
#                               axis=0)

# def ClaytonCDF(x, theta1=1., theta2=1. / 2, scale1=1., scale2=1. / 2, par=1.5):
#
#     if (x[0] > 0. and x[1] > 0.):
#
#         x1 = 1 - ((1 + theta1 * x[0] / scale1) ** -(1 / theta1))
#         x2 = 1 - ((1 + theta2 * x[1] / scale2) ** -(1 / theta2))
#         return (x1 ** -par + x2 ** -par - 1) ** -(1 / par)
#     else:
#         return 0.

##### Clayton #####

class ClaytonCDF:
    """
    Frank cumulative distribution function.

    :param par: Frank copula parameter.
    :type par: float

    """
    def __init__(self,par):
        self.par =par
    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, var):
        if var < 0:
            raise ValueError("The input 'par' must be non-negative")
        self.__par = var

    def cdf(self,x):
        """
        :param x: Array with shape (N,d) where N is the number of points and d the dimension
        :type x: numpy.ndarray

        :return: Frank c.d.f. for each row of x.
        :rtype: numpy.ndarray

        """
        if len(x.shape) == 1:
            if (x <= 0).any():
                return 0
            else:
                return (np.sum(np.minimum(x, 1) ** (-self.par) - 1) + 1) ** -(1 / self.par)
        N = len(x)
        output = np.array([0.] * N)
        PosIndex = ~np.array(x <= 0).any(axis=1)
        output[PosIndex] = (np.sum(np.minimum(x[PosIndex, :], 1) ** (-self.par) - 1, axis=1) + 1) ** (-1 / self.par)
        return output

##### Frank #####
class FrankCDF:
    """
    Frank cumulative distribution function.

    :param par: Frank copula parameter.
    :type par: float

    """

    def __init__(self, par):
        self.par = par

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, var):
        if var < 0:
            raise ValueError("The input 'par' must be non-negative")
        self.__par = var

    def cdf(self, x):
        """
        :param x: Array with shape (N,d) where N is the number of points and d the dimension
        :type x: numpy.ndarray

        :return: Frank c.d.f. for each row of x.
        :rtype: numpy.ndarray

        """
        if len(x.shape) == 1:
            if (x <= 0).any():
                return 0
            else:
                d = len(x)
                return -1 / self.par * np.log(
                    1 + np.prod(np.exp(-self.par * np.minimum(x, 1)) - 1) / (np.exp(-self.par) - 1) ** (d - 1))
        N = len(x)
        d = len(x[0])
        output = np.array([0.] * N)
        PosIndex = ~np.array(x <= 0).any(axis=1)
        output[PosIndex] = -1 / self.par * np.log(
            1 + np.prod(np.exp(-self.par * np.minimum(x[PosIndex, :], 1)) - 1, axis=1) / (np.exp(-self.par) - 1) ** (d - 1))
        return output

# def FrankCDF(x, par):
#     """
#     Frank cumulative distribution function.
#
#     :param x: Array with shape (N,d) where N is the number of points and d the dimension
#     :type x: numpy.ndarray
#     :param par: Frank copula parameter.
#     :type par: float
#
#     :return: Frank c.d.f. for each row of x.
#     :rtype: numpy.ndarray
#     """
#     if par < 0:
#         raise ValueError("The input 'par' must be non-negative")
#     if len(x.shape) == 1:
#         if (x <= 0).any():
#             return 0
#         else:
#             d = len(x)
#             return -1 / par * np.log(1 + np.prod(np.exp(-par * np.minimum(x, 1)) - 1) / (np.exp(-par) - 1) ** (d - 1))
#     N = len(x)
#     d = len(x[0])
#     output = np.array([0.] * N)
#     PosIndex = ~np.array(x <= 0).any(axis=1)
#     output[PosIndex] = -1 / par * np.log(
#         1 + np.prod(np.exp(-par * np.minimum(x[PosIndex, :], 1)) - 1, axis=1) / (np.exp(-par) - 1) ** (d - 1))
#     return output


##### Gumbel #####

## GumbelCDF
class GumbelCDF:
    """
    Gumbel cumulative distribution function.

    :param par: Gumbel copula parameter.
    :type par: float

    """

    def __init__(self, par):
        self.par = par

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, var):
        if var < 0:
            raise ValueError("The input 'par' must be non-negative")
        self.__par = var

    def cdf(self, x):
        """
        :param x: Array with shape (N,d) where N is the number of points and d the dimension
        :type x: numpy.ndarray

        :return: Gumbel c.d.f. for each row of x.
        :rtype: numpy.ndarray

        """
        if len(x.shape) == 1:
            if (x <= 0).any():
                return 0
            else:
                return np.exp(-np.sum((-np.log(np.minimum(x, 1))) ** self.par) ** (1 / self.par))
        N = len(x)
        output = np.array([0.] * N)
        PosIndex = ~np.array(x <= 0).any(axis=1)
        output[PosIndex] = np.exp(-np.sum((-np.log(np.minimum(x[PosIndex, :], 1))) ** self.par, axis=1) ** (1 / self.par))
        return output


# def GumbelCDF(x, par):
#     """
#     Gumbel cumulative distribution function.
#
#     :param x: Array with shape (N,d) where N is the number of points and d the dimension
#     :type x: numpy.ndarray
#     :param par: Gumbel copula parameter.
#     :type par: float
#
#     :return: Gumbel c.d.f. for each row of x.
#     :rtype: numpy.ndarray
#     """
#     if par < 1:
#         raise ValueError("The input 'par' must be non-negative")
#     if len(x.shape) == 1:
#         if (x <= 0).any():
#             return 0
#         else:
#             return np.exp(-np.sum((-np.log(np.minimum(x, 1))) ** par) ** (1 / par))
#     N = len(x)
#     output = np.array([0.] * N)
#     PosIndex = ~np.array(x <= 0).any(axis=1)
#     output[PosIndex] = np.exp(-np.sum((-np.log(np.minimum(x[PosIndex, :], 1))) ** par, axis=1) ** (1 / par))
#     return output

### Marginals ###
class MarginalCdf2D:
    def __init__(self,dist1,dist2,d_la):
        self.dist1=dist1
        self.dist2=dist2
        self.d_la=d_la

    def marginal_cdf(self,x):

        if len(x.shape) == 1:
            return np.array([self.dist1.cdf(x[0]), self.dist2.cdf(x[1])])

        out_= np.zeros(x.shape[0]*x.shape[1]).reshape(x.shape[0],-1)
        out_[:,0]= self.dist1.cdf(x[:,0])
        out_[:,1] = self.dist2.cdf(x[:,1])
        return out_

class MarginalCdf3D:
    def __init__(self,
                 dist1,
                 dist2,
                 dist3,
                 d_la):
        self.dist1=dist1
        self.dist2=dist2
        self.dist3=dist3
        self.d_la=d_la

    def marginal_cdf(self,x):
        if len(x.shape) == 1:
            return np.array([self.dist1.cdf(x[0]), self.dist2.cdf(x[1]),self.dist3.cdf(x[2])])

        out_= np.zeros(x.shape[0]*x.shape[1]).reshape(x.shape[0],-1)
        out_[:,0]= self.dist1.cdf(x[:,0])
        out_[:,1] = self.dist2.cdf(x[:,1])
        out_[:,2] = self.dist3.cdf(x[:,2])
        return out_

class MarginalCdf4D:
    def __init__(self,
                 dist1,
                 dist2,
                 dist3,
                 dist4,
                 d_la):
        self.dist1=dist1
        self.dist2=dist2
        self.dist3=dist3
        self.dist4=dist4
        self.d_la=d_la

    def marginal_cdf(self,x):
        if len(x.shape) == 1:
            return np.array([self.dist1.cdf(x[0]),
                             self.dist2.cdf(x[1]),
                             self.dist3.cdf(x[2]),
                             self.dist4.cdf(x[3])])

        out_= np.zeros(x.shape[0]*x.shape[1]).reshape(x.shape[0],-1)
        out_[:,0]= self.dist1.cdf(x[:,0])
        out_[:,1] = self.dist2.cdf(x[:,1])
        out_[:,2] = self.dist3.cdf(x[:,2])
        out_[:,3] = self.dist4.cdf(x[:,3])
        return out_

class MarginalCdf5D:
    def __init__(self,
                 dist1,
                 dist2,
                 dist3,
                 dist4,
                 dist5,
                 d_la):

        self.dist1=dist1
        self.dist2=dist2
        self.dist3=dist3
        self.dist4=dist4
        self.dist5 = dist5
        self.d_la=d_la

    def marginal_cdf(self,x):
        if len(x.shape) == 1:
            return np.array([self.dist1.cdf(x[0]),
                             self.dist2.cdf(x[1]),
                             self.dist3.cdf(x[2]),
                             self.dist4.cdf(x[3]),
                             self.dist5.cdf(x[3])])

        out_= np.zeros(x.shape[0]*x.shape[1]).reshape(x.shape[0],-1)
        out_[:,0]= self.dist1.cdf(x[:,0])
        out_[:,1] = self.dist2.cdf(x[:,1])
        out_[:,2] = self.dist3.cdf(x[:,2])
        out_[:,3] = self.dist4.cdf(x[:,3])
        out_[:,4] = self.dist5.cdf(x[:,4])
        return out_

# def ClaytonCDF(x, par=1.5,d=2):
#     """
#     Clayton cumulative distribution function.
#
#     :param x: vettore
#     :type x: numpy.ndarray
#     :param par: Clayton parameter.
#     :type par: float
#     :param d: Copula dimension.
#     :type d: int
#
#     :return: clayton cdf
#     :rtype: numpy.ndarray
#     """
#     if (x>0.).all():
#         if (x == 1.).any():
#             return 1.
#         else:
#             return (np.sum(x**-par)-d+1) ** -(1 / par)
#         # return (x[0] ** -par + x[1] ** -par - 1) ** -(1 / par)
#     else:
#         return 0.
#
# def GumbelCDF(x, par=1):
#
#     if (x > 0.).all():
#         if (x == 1.).any():
#             return 1.
#         else:
#             return np.exp(-(np.sum((-np.log(x)) ** par)**(1 / par)))
#     else:
#         return 0.

#aggiungi la copula
def Cop(x): #input matrice le cui colonne sono punti.
    return np.apply_along_axis(func1d=ClaytonCDF,arr=x,axis=0)

def v_h(b, h, Copula, Mat):
    # alpha= d/(d+1)
    s = (-1) ** np.sum(Mat, axis=0)
    return sum(Copula(b.reshape(2, 1) + h * Mat) * s)

def lrcrm_f1(x,dist):
    """
    It simulates a random number from a poisson distribution.
    It simulates a random number from a distribution a poisson distribution with parameter mu.

    :param x: distribution parameter.
    :type x: float
    :param dist: poisson distribution.
    :type dist: scipy.stats._discrete_distns.poisson_gen

    :return:simulated random number.
    :rtype: numpy.ndarray
    """
    return dist(mu=x).rvs(1)

def lrcrm_f2(x,dist):
    """
    It simulates random values from a gamma.

    Parameters:
    :param x: it contains the gamma parameters and the number of random values to be simulated.
    :type x: numpy.ndarray
    :param dist: gamma distribution.
    :type dist: scipy.stats._discrete_distns.gamma_gen

    :return: sum of the simulated numbers.
    :rtype: numpy.ndarray
    """
    return np.sum(dist(a=x[1],scale=x[2]).rvs(int(x[0])))
