import math
import numpy as np

def mean_custom(data):
    """
    Computes the arithmetic mean (1st raw moment).
    """
    data = np.asarray(data)
    return np.sum(data) / len(data)

def variance_custom(data, ddof=1):
    """
    Computes the variance: the unbiased sample estimator (ddof=1, Bessel's
    correction) by default, or the population 2nd central moment (ddof=0).
    """
    data = np.asarray(data)
    mu = mean_custom(data)
    return np.sum((data - mu) ** 2) / (len(data) - ddof)

def std_dev_custom(data, ddof=1):
    """
    Computes the standard deviation.
    """
    return np.sqrt(variance_custom(data, ddof))

def skewness_custom(data):
    """
    Computes the skewness of the distribution (standardized 3rd central moment).
    """
    data = np.asarray(data)
    mu = mean_custom(data)
    sigma = np.std(data, ddof=0)  # population standard deviation, used for central moments
    if sigma == 0.0:
        return 0.0
    return np.mean(((data - mu) / sigma) ** 3)

def kurtosis_custom(data):
    """
    Computes the raw kurtosis of the distribution (standardized 4th central
    moment, not excess kurtosis). For a normal distribution, the theoretical
    value is 3.0 (excess kurtosis = raw kurtosis - 3).
    """
    data = np.asarray(data)
    mu = mean_custom(data)
    sigma = np.std(data, ddof=0)  # population standard deviation, used for central moments
    if sigma == 0.0:
        return 0.0
    return np.mean(((data - mu) / sigma) ** 4)

def binomial_pmf(k, n, p):
    """
    Probability Mass Function (PMF) of the Binomial distribution.
    Measures the probability of k successes in n trials with success probability p.
    """
    if k < 0 or k > n:
        return 0.0
    if not (0.0 <= p <= 1.0):
        raise ValueError("Probability p must be in the [0, 1] interval")
    comb = math.comb(n, k)
    return comb * (p ** k) * ((1.0 - p) ** (n - k))

def poisson_pmf(k, lam):
    """
    Probability Mass Function (PMF) of the Poisson distribution.
    Measures the probability of k events in a fixed interval with mean rate lam.
    """
    if k < 0:
        return 0.0
    if lam <= 0.0:
        raise ValueError("The lambda (rate) parameter must be positive")
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def normal_pdf(x, mu, sigma):
    """
    Probability Density Function (PDF) of the Normal (Gaussian) distribution.
    """
    if sigma <= 0.0:
        raise ValueError("The standard deviation sigma must be strictly positive")
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = math.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))
    return coef * exponent

def fit_normal_mle(data):
    """
    Estimates the mu and sigma parameters of a Gaussian distribution from the
    data using Maximum Likelihood Estimation (MLE).
    """
    data = np.asarray(data)
    mu = mean_custom(data)
    # The maximum-likelihood variance estimator is biased (divided by n, ddof=0)
    sigma = np.std(data, ddof=0)
    return mu, sigma
