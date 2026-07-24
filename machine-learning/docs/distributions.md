# 🎲 Statistical Moments and Probability Distributions

This document describes the theoretical-moment and probability-function formulations implemented in the `src/distributions.py` module, using Bishop's (PRML) notation.

---

## 1. Statistical Moments
Let $x$ be a one-dimensional real random variable with a set of observations $\{x_n\}_{n=1}^N$.

### Arithmetic Mean ($\mu$ - First Raw Moment)
Represents the central value of the distribution and corresponds to the first raw moment $\mathbb{E}[x]$:

$$\mu = \frac{1}{N} \sum_{n=1}^N x_n$$

### Variance ($\sigma^2$ - Second Central Moment)
Measures the dispersion of the data around the mean. It corresponds to the second central moment $\mathbb{E}[(x - \mu)^2]$. In practice we compute one of two estimators, controlled by the degrees-of-freedom parameter `ddof`:

$$\sigma^2 = \frac{1}{N - d} \sum_{n=1}^N (x_n - \mu)^2$$

*   $d = 0$ gives the **population variance** (equivalently, the biased maximum-likelihood estimator) — this is the true second central moment.
*   $d = 1$ gives the **unbiased sample variance estimator** (Bessel's correction), which is *not* itself a raw central moment but corrects for the bias introduced by estimating $\mu$ from the same sample. This is the default used throughout `src/distributions.py`.

### Skewness ($\gamma_1$)
Measures the asymmetry of the probability distribution relative to its mean. It is the standardized third central moment, computed here with the population standard deviation ($d=0$):

$$\gamma_1 = \mathbb{E}\left[ \left( \frac{x - \mu}{\sigma} \right)^3 \right] = \frac{\frac{1}{N} \sum_{n=1}^N (x_n - \mu)^3}{\sigma^3}$$

### Kurtosis ($\beta_2$)
Measures the tail weight/peakedness of the distribution. It is the standardized fourth central moment, likewise computed with the population standard deviation:

$$\beta_2 = \mathbb{E}\left[ \left( \frac{x - \mu}{\sigma} \right)^4 \right] = \frac{\frac{1}{N} \sum_{n=1}^N (x_n - \mu)^4}{\sigma^4}$$

For a pure normal distribution, $\beta_2 = 3.0$. Note this is the **raw kurtosis**, not *excess* kurtosis (a common source of ambiguity): excess kurtosis is defined as $\beta_2 - 3$, which is $0$ for the normal distribution.

---

## 2. Probability Distribution Functions

### Binomial Distribution
Let $k$ be the number of successes obtained in $n$ independent Bernoulli trials with success probability $p$:

$$\text{Bin}(k \mid n, p) = \binom{n}{k} p^k (1 - p)^{n-k}$$

Where the binomial coefficient is defined as:
$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

### Poisson Distribution
Describes the probability of a number $k$ of events occurring in a fixed interval of time or space, with mean rate $\lambda$:

$$\text{Poi}(k \mid \lambda) = \frac{e^{-\lambda} \lambda^k}{k!}$$

### Normal (Gaussian) Distribution
The probability density of the continuous variable $x$ parameterized by mean $\mu$ and variance $\sigma^2$ is given by:

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( - \frac{(x - \mu)^2}{2\sigma^2} \right)$$

---

## 3. Maximum Likelihood Estimation (MLE)
Given an independent and identically distributed (i.i.d.) sample of observations $\mathbf{X} = \{x_1, \dots, x_N\}$, the likelihood function for a Gaussian distribution is:

$$p(\mathbf{X} \mid \mu, \sigma^2) = \prod_{n=1}^N \mathcal{N}(x_n \mid \mu, \sigma^2)$$

Maximizing the log-likelihood with respect to $\mu$ and $\sigma^2$ yields the maximum-likelihood estimators:

$$\mu_{\text{ML}} = \frac{1}{N} \sum_{n=1}^N x_n$$

$$\sigma^2_{\text{ML}} = \frac{1}{N} \sum_{n=1}^N (x_n - \mu_{\text{ML}})^2$$

Note that $\sigma^2_{\text{ML}}$ is exactly the population variance ($d=0$ above) — the maximum-likelihood variance estimator is biased, which is why the unbiased sample estimator ($d=1$, Bessel's correction) is preferred when estimating dispersion from a sample rather than fitting a distribution by MLE.

---

## 4. Pure Python Implementation (From Scratch)
Straight from `src/distributions.py`, with no dependency on `scipy`:

```python
import math
import numpy as np

def mean_custom(data):
    data = np.asarray(data)
    return np.sum(data) / len(data)

def variance_custom(data, ddof=1):
    data = np.asarray(data)
    mu = mean_custom(data)
    return np.sum((data - mu) ** 2) / (len(data) - ddof)

def std_dev_custom(data, ddof=1):
    return np.sqrt(variance_custom(data, ddof))

def skewness_custom(data):
    data = np.asarray(data)
    mu = mean_custom(data)
    sigma = np.std(data, ddof=0)
    if sigma == 0.0:
        return 0.0
    return np.mean(((data - mu) / sigma) ** 3)

def kurtosis_custom(data):
    data = np.asarray(data)
    mu = mean_custom(data)
    sigma = np.std(data, ddof=0)
    if sigma == 0.0:
        return 0.0
    return np.mean(((data - mu) / sigma) ** 4)

def binomial_pmf(k, n, p):
    comb = math.comb(n, k)
    return comb * (p ** k) * ((1.0 - p) ** (n - k))

def poisson_pmf(k, lam):
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def normal_pdf(x, mu, sigma):
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = math.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))
    return coef * exponent

def fit_normal_mle(data):
    data = np.asarray(data)
    mu = mean_custom(data)
    sigma = np.std(data, ddof=0)  # biased ML estimator
    return mu, sigma
```

---

## 5. Library Implementation (scipy)

```python
import numpy as np
from scipy.stats import binom, poisson, norm

samples_binom = binom.rvs(n=10, p=0.5, size=1000)
mean, var, skew, kurt = binom.stats(n=10, p=0.5, moments='mvsk')
p_k = binom.pmf(k=5, n=10, p=0.5)

p_k_poisson = poisson.pmf(k=3, mu=2.5)

samples_norm = norm.rvs(loc=0.0, scale=1.0, size=1000)
density_x = norm.pdf(x=0.0, loc=0.0, scale=1.0)

# MLE fit
mu_hat, sigma_hat = norm.fit(samples_norm)
```

---

## 6. Worked Example (By Hand)

### 6.1 Binomial PMF
$n=5$ trials, $p=0.5$, probability of exactly $k=3$ successes:
$$\binom{5}{3} = \frac{5!}{3!\,2!} = 10, \qquad P(X=3) = 10 \times 0.5^3 \times 0.5^2 = 10 \times 0.125 \times 0.25 = 0.3125\ (31.25\%)$$

### 6.2 Poisson PMF
Mean rate $\lambda=2$, probability of $k=3$ events:
$$P(X=3) = \frac{e^{-2}\,2^3}{3!} = \frac{0.1353 \times 8}{6} \approx 0.1804\ (18.04\%)$$

### 6.3 Normal PDF
Standard normal ($\mu=0,\sigma=1$), density at $x=1$:
$$f(1) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}} = 0.3989 \times 0.6065 \approx 0.2420$$

### 6.4 Mean, MLE Variance, and Unbiased Sample Variance
Sample $\{2, 4, 6\}$:
$$\mu = \frac{2+4+6}{3} = 4$$
$$\sigma^2_{\text{ML}} = \frac{(2-4)^2+(4-4)^2+(6-4)^2}{3} = \frac{4+0+4}{3} = 2.667 \quad (\text{ddof}=0)$$
$$s^2 = \frac{(2-4)^2+(4-4)^2+(6-4)^2}{3-1} = \frac{8}{2} = 4.0 \quad (\text{ddof}=1,\text{ Bessel-corrected})$$
As expected, the biased MLE estimate ($2.667$) is smaller than the unbiased sample estimate ($4.0$).
