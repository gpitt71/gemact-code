"""
This script contains helper functions to be used in the main scripts.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cholesky
from scipy.special import gammaincinv
import scipy.stats as ss

def arg_type_handler(x):
    type_tuple = (float, int, list, np.ndarray)
    assert isinstance(x, type_tuple), 'x should be one of %r' %type_tuple
    if type(x) == list:
        x = np.array(x)
    if type(x) == np.ndarray:
        x = x.flatten()
    if type(x) == int or type(x) == float:
        x = np.array([x])
    return x

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

## LossReserve
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

def lrcrm_f1(x, dist):
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

def lrcrm_f2(x, dist):
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

def cartesian_product(*arrays):
    """
    Generates the matrix points where copula is computed.

    :param d: dimension.
    :type d: ``int``
    :return: matrix of points.
    :rtype:``numpy.ndarray``
    """
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)

    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows)

def cov_to_corr(cov):
    """
    Covariance matrix to correlation matrix converter.
    Equivalent to R 'cov2corr' function.
    """
    d = np.sqrt(cov.diagonal())
    corr = ((cov.T/cov).T)/d
    return corr

def multivariate_t_cdf(x, corr, df, tol, max_evaluations, n_repetitions):
    """
    Function for estimating the cdf of a multivariate t distribution using quasi-monte carlo algorithm.
    See:
    - Genz, A. and F. Bretz (1999) "Numerical Computation of Multivariate
    t Probabilities with Application to Power Calculation of Multiple
    Contrasts", J.Statist.Comput.Simul., 63:361-378.
    - Genz, A. and F. Bretz (2002) "Comparison of Methods for the
    Computation of Multivariate t Probabilities", J.Comp.Graph.Stat.,
    11(4):950-971.

    Parameters
    ----------
    x : array_like
        Standardized sample, shape (n_features,)
    corr : array_like, shape (n_features, n_features)
        Correlation matrix of the distribution,must be symmetric and positive
        definite, with all elements of the diagonal being 1
    df : float
        Degrees-of-freedom of the distribution, must be a positive real number
    tol: float
        Tolerance for quasi-Monte Carlo algorithm
    max_evaluations: float
        Maximum points to evaluate with quasi-Monte Carlo algorithm
    n_repetitions: int
        Number of repetitions of quasi-Monte Carlo algorithm

    Returns
    -------
    cdf: scalar
        Cumulative density function evaluated at `x`
    err: scalar
        Error estimate of cdf numerical approximation
    """

    if np.all(x == np.inf):
        return (1, 0)
    if np.all(x == -np.inf):
        return (0, 0)

    chol = cholesky(corr)

    primes = [31, 47, 71, 107, 163,
    241, 359, 541, 811, 1217,
    1823, 2731, 4099, 6151, 9227,
    13841, 20759, 31139, 46703, 70051,
    105071, 157627, 236449, 354677, 532009,
    798023, 1197037, 1795559, 2693329, 4039991,
    6059981, 9089981]
    # max_prime = 100000 # move max_prime as function argument?
    # primes = primesfrom2to(max_prime)

    x = x / np.diag(chol)
    chol = chol / np.tile(np.diag(chol), (chol.shape[0], 1))

    fun_evaluations = 2 * n_repetitions * primes[0]
    t, sigma_squared = _multivariate_t_cdf_qmc(x, chol, df, n_repetitions, primes[0])
    err = 3.5 * np.sqrt(sigma_squared)

    for prime in primes[1:]:
        fun_evaluations += 2 * n_repetitions * prime
        if fun_evaluations > max_evaluations:
            break

        t_hat, sigma_squared_tau_hat = _multivariate_t_cdf_qmc(x, chol, df, n_repetitions, prime)

        t += sigma_squared * (t_hat - t) / (sigma_squared + sigma_squared_tau_hat)
        sigma_squared *= sigma_squared_tau_hat / (sigma_squared_tau_hat + sigma_squared)
        err = 3.5 * np.sqrt(sigma_squared)

        if err < tol:
            return t, err
    return (t, err)

def _multivariate_t_cdf_qmc(x, chol, df, mc_repetitions, prime):
    sampler = ss.qmc.Halton(d=chol.shape[0], scramble=True)
    p = sampler.random(n=prime)
    t_hat = np.zeros(mc_repetitions)

    for rep in range(mc_repetitions):
        w = np.tile(np.random.uniform(low=0.0, high=1.0, size=chol.shape[0]), (prime, 1))
        v = abs(2 * ((p + w) % 1) - 1)

        # f = lambda w: _t_estimate(x, chol, df, w, prime, chol.shape[0])
        t_hat[rep] = 0.5*(_t_estimate(x, chol, df, v, prime, chol.shape[0]) +\
            _t_estimate(x, chol, df, 1-v, prime, chol.shape[0]))
        # t_hat[rep] = f(np.array([v, 1-v])).mean()

    return np.mean(t_hat), np.var(t_hat) / mc_repetitions

def _t_estimate(x, chol, df, v, n, d):
    eps_tolerance = np.finfo(float).eps
    s = (2 * gammaincinv(df / 2, v[:, -1])) ** 0.5 / (df ** 0.5)
    e = ss.norm.cdf(s * x[0])
    t = 1.0 * e
    y = np.zeros((n, d))

    for i in range(1, d):
        z = np.maximum(np.minimum(e * v[:, -1 - i], 1 - eps_tolerance), eps_tolerance)
        y[:, i - 1] = ss.norm.ppf(z)
        e = ss.norm.cdf(s * x[i] - np.dot(y, chol[:, i]))
        t *= e
    return t.mean()