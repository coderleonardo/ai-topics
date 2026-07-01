import math
import numpy as np

def mean_custom(data):
    """
    Calcula a média aritmética (1º momento bruto).
    """
    data = np.asarray(data)
    return np.sum(data) / len(data)

def variance_custom(data, ddof=1):
    """
    Calcula a variância amostral (ddof=1) ou populacional (ddof=0) (2º momento central).
    """
    data = np.asarray(data)
    mu = mean_custom(data)
    return np.sum((data - mu) ** 2) / (len(data) - ddof)

def std_dev_custom(data, ddof=1):
    """
    Calcula o desvio padrão.
    """
    return np.sqrt(variance_custom(data, ddof))

def skewness_custom(data):
    """
    Calcula a assimetria (skewness) da distribuição (3º momento central padronizado).
    """
    data = np.asarray(data)
    mu = mean_custom(data)
    sigma = np.std(data, ddof=0) # Desvio padrão populacional para momentos centrais
    if sigma == 0.0:
        return 0.0
    return np.mean(((data - mu) / sigma) ** 3)

def kurtosis_custom(data):
    """
    Calcula a curtose (kurtosis) bruta da distribuição (4º momento central padronizado).
    Para uma distribuição normal, o valor teórico é 3.0.
    """
    data = np.asarray(data)
    mu = mean_custom(data)
    sigma = np.std(data, ddof=0) # Desvio padrão populacional para momentos centrais
    if sigma == 0.0:
        return 0.0
    return np.mean(((data - mu) / sigma) ** 4)

def binomial_pmf(k, n, p):
    """
    Função de Massa de Probabilidade (PMF) da distribuição Binomial.
    Mede a probabilidade de k sucessos em n tentativas com probabilidade p de sucesso.
    """
    if k < 0 or k > n:
        return 0.0
    if not (0.0 <= p <= 1.0):
        raise ValueError("A probabilidade p deve estar no intervalo [0, 1]")
    comb = math.comb(n, k)
    return comb * (p ** k) * ((1.0 - p) ** (n - k))

def poisson_pmf(k, lam):
    """
    Função de Massa de Probabilidade (PMF) da distribuição de Poisson.
    Mede a probabilidade de k eventos num intervalo fixo com taxa média lam.
    """
    if k < 0:
        return 0.0
    if lam <= 0.0:
        raise ValueError("O parâmetro lambda (taxa) deve ser positivo")
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def normal_pdf(x, mu, sigma):
    """
    Função Densidade de Probabilidade (PDF) da distribuição Normal (Gaussiana).
    """
    if sigma <= 0.0:
        raise ValueError("O desvio padrão sigma deve ser estritamente positivo")
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = math.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))
    return coef * exponent

def fit_normal_mle(data):
    """
    Estima os parâmetros mu e sigma de uma distribuição Gaussiana dados os dados
    usando o Estimador de Máxima Verossimilhança (MLE).
    """
    data = np.asarray(data)
    mu = mean_custom(data)
    # O estimador de máxima verossimilhança de variância é enviesado (dividido por n, ddof=0)
    sigma = np.std(data, ddof=0)
    return mu, sigma
