"""
This script contains helper functions to be used in the main scripts.
"""

from .libraries import *


def arg_type_handler(x):
    """
    Checks that arguments in distribution.py methods are managed correctly.

    :param x: method input
    :type x: any

    :return: x
    :rtype: ``numpy.ndarray``
    """
    type_tuple = (float, int, list, np.ndarray)
    assert isinstance(x, type_tuple), "x should be one of %r" % (type_tuple,)
    if type(x) == list:
        x = np.array(x)
    if type(x) == np.ndarray:
        x = x.flatten()
    if type(x) == int or type(x) == float:
        x = np.array([x])
    return x


def ecdf(x):
    """
    Empirical cumulative density function computed on the vector x.
    Used in the lossmodel.py script.

    :param x:  sequence of values to compute the ecdf on.
    :type x: ``numpy.ndarray``

    :return: starting sequence, empirical cumulative density function.
    :rtype: ``numpy.ndarray``
    """
    dim = len(x)
    x_ = np.sort(x)
    y_ = np.cumsum(np.repeat(1, dim)) / dim
    f = interpolate.interp1d(x_, y_)

    return x_, f(x_)


# LossReserve
def normalizernans(x):
    """
    Normalize a vector with nan values ignoring the nan values during the computation.
    Used in the lossreserve.py script.

    :param x: sequence to be normalized.
    :type x: ``numpy.ndarray``

    :return: normalized sequence.
    :rtype: ``numpy.ndarray``

    """
    if np.sum(np.isnan(x)) < x.shape[0]:
        x[~np.isnan(x)] = x[~np.isnan(x)]/np.sum(x[~np.isnan(x)])
    return x


def lrcrm_f1(x, dist):
    """
    Simulate a random number from a distribution a poisson distribution with parameter mu.
    Used in the lossreserve.py script.

    :param x: distribution parameter.
    :type x: ``float``
    :param dist: poisson distribution.
    :type dist: ``scipy.stats._discrete_distns.poisson_gen``

    :return:simulated random number.
    :rtype: ``numpy.ndarray``
    """
    return dist(mu=x).rvs(1)


def lrcrm_f2(x, dist):
    """
    Simulates random values from a gamma.
    Used in the lossreserve.py script.

    :param x: it contains the gamma parameters and the number of random values to be simulated.
    :type x: ``numpy.ndarray``
    :param dist: gamma distribution.
    :type dist: ``scipy.stats._discrete_distns.gamma_gen``

    :return: sum of the simulated numbers.
    :rtype: ``numpy.ndarray``
    """
    return np.sum(dist(a=x[1], scale=x[2]).rvs(int(x[0])))


# Loss aggregation
def cartesian_product(*arrays):
    """
    Generate the matrix points where copula is computed.
    Used in the lossaggregation.py script.

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

    Used in the copulas.py script.

    :param cov: matrix of covariates
    :type cov: ``numpy.ndarray``

    :return: matrix of correlations.
    :rtype: ``numpy.ndarray``
    """
    d = np.sqrt(cov.diagonal())
    corr = (cov.T/cov).T/d
    return corr


def multivariate_t_cdf(x, corr, df, tol, iterations):
    """
    Estimate the cdf of a multivariate t distribution using quasi-Monte Carlo algorithm.
    Used in the copulas.py script.

    See:
    - Genz, A. and F. Bretz (1999) "Numerical Computation of Multivariate
    t Probabilities with Application to Power Calculation of Multiple
    Contrasts", J.Statist.Comput.Simul., 63:361-378.
    - Genz, A. and F. Bretz (2002) "Comparison of Methods for the
    Computation of Multivariate t Probabilities", J.Comp.Graph.Stat.,
    11(4):950-971.

    :param x: quantile where the cumulative distribution function is evaluated.
    :type x: ``numpy.ndarray``
    :param corr: correlation matrix of the distribution,must be symmetric and positive
                 definite, with all elements of the diagonal being 1.
    :type corr: ``numpy.ndarray``
    :param df: degrees-of-freedom of the distribution, must be a positive real number.
    :type df: ``float``
    :param tol: tolerance for quasi-Monte Carlo algorithm.
    :type tol: ``float``
    :param iterations: number of iterations of quasi-Monte Carlo algorithm.
    :type iterations: ``int``

    :return: cumulative density function value and error estimate of the numerical approximation.
    :rtype: ``tuple``
    """

    if np.all(x == np.inf):
        return tuple((1, 0))
    if np.all(x == -np.inf):
        return tuple((0, 0))

    chol = cholesky(corr)

    primes = [31, 47, 71, 107, 163,
              241, 359, 541, 811, 1217,
              1823, 2731, 4099, 6151, 9227,
              13841, 20759, 31139, 46703, 70051,
              105071, 157627, 236449, 354677, 532009,
              798023, 1197037, 1795559, 2693329, 4039991,
              6059981, 9089981]

    x = x / np.diag(chol)
    chol = chol / np.tile(np.diag(chol), (chol.shape[0], 1))

    t, sigma_squared = _multivariate_t_cdf_qmc(x, chol, df, iterations, primes[0])
    err = 3.5 * np.sqrt(sigma_squared)

    for prime in primes[1:]:
        t_hat, sigma_squared_tau_hat = _multivariate_t_cdf_qmc(x, chol, df, iterations, prime)

        t += sigma_squared * (t_hat - t) / (sigma_squared + sigma_squared_tau_hat)
        sigma_squared *= sigma_squared_tau_hat / (sigma_squared_tau_hat + sigma_squared)
        err = 3.5 * np.sqrt(sigma_squared)

        if err < tol:
            return t, err

    return tuple((t, err))


def _multivariate_t_cdf_qmc(x, chol, df, iterations, size):
    """
    Multivariate t cumulative density function computed via quasi-Monte Carlo.
    References:
    - Genz, A. and F. Bretz (1999) "Numerical Computation of Multivariate t Probabilities with Application to Power Calculation of Multiple Contrasts", Journal of Statistical Computation and Simulation, 63:361-378.
    - Genz, A. and F. Bretz (2002) "Comparison of Methods for the Computation of Multivariate t Probabilities",  Journal of Computational and Graphical Statistics, 11(4):950-971.

    :param x: quantile where the cumulative distribution function is evaluated.
    :type x: ``numpy.ndarray``
    :param chol: Cholesky decomposition of the correlation matrix of the distribution, must be symmetric and positive
                 definite, with all elements of the diagonal being 1.
    :type chol: ``numpy.ndarray``
    :param df: degrees-of-freedom of the distribution.
    :type df: float
    :param iterations: number of iterations.
    :type iterations: ``int``
    :param size: sample size.
    :type size: ``int``

    :return: quasi-Monte Carlo estimate of the t distribution cumulative density function.
    :rtype: ``float``
    """

    sampler = stats.qmc.Halton(d=chol.shape[0], scramble=True)
    p = sampler.random(n=size)
    t_ = np.zeros(iterations)

    for rep in range(iterations):
        w = np.tile(np.random.uniform(low=0.0, high=1.0, size=chol.shape[0]), (size, 1))
        w = abs(2 * ((p + w) % 1) - 1)

        t_[rep] = 0.5*(_t_separation_variable(x, chol, df, w, size) +
                    _t_separation_variable(x, chol, df, 1-w, size))

    return np.mean(t_), np.var(t_) / iterations


def _t_separation_variable(x, chol, df, w, size):
    """
    Separation of variables transformation helper function to estimate t distribution cumulative density function.

    :param x: quantile where the cumulative distribution function is evaluated.
    :type x: ``numpy.ndarray``
    :param chol: Cholesky decomposition of the correlation matrix of the distribution, must be symmetric and positive
                 definite, with all elements of the diagonal being 1.
    :type chol: ``numpy.ndarray``
    :param df: degrees-of-freedom of the distribution.
    :type df: ``float``
    :param w: transformed uniform variable.
    :type w: ``float``
    :param size: sample size.
    :type size: ``int``

    :return: helper estimate of the t distribution cumulative density function.
    :rtype: ``float``
    """
    
    eps_tolerance = np.finfo(float).eps
    gam_inv = (2 * special.gammaincinv(df / 2, w[:, -1])) ** 0.5 / (df ** 0.5)
    norm_prob = stats.norm.cdf(gam_inv * x[0])
    norm_quant = np.zeros((size, chol.shape[0]))

    t_separate_hat = norm_prob
    for i in range(1, chol.shape[0]):
        max_min = np.maximum(np.minimum(norm_prob * w[:, -1 - i], 1 - eps_tolerance), eps_tolerance)
        norm_quant[:, i - 1] = stats.norm.ppf(max_min)
        norm_prob = stats.norm.cdf(gam_inv * x[i] - np.dot(norm_quant, chol[:, i]))
        t_separate_hat *= norm_prob
    return np.mean(t_separate_hat)