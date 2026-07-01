# 🎲 Momentos Estatísticos e Distribuições de Probabilidade

Este documento descreve as formulações de momentos teóricos e funções de probabilidade implementadas no módulo `src/distributions.py` com a notação de Bishop (PRML).

---

## 1. Momentos Estatísticos
Seja $x$ uma variável aleatória real unidimensional com um conjunto de observações $\{x_n\}_{n=1}^N$.

### Média Aritmética ($\mu$ - Primeiro Momento Bruto)
Representa o valor central da distribuição e corresponde ao primeiro momento bruto $\mathbb{E}[x]$:

$$\mu = \frac{1}{N} \sum_{n=1}^N x_n$$

### Variância ($\sigma^2$ - Segundo Momento Central)
Mede a dispersão dos dados em relação à média. Corresponde ao segundo momento central $\mathbb{E}[(x - \mu)^2]$:

$$\sigma^2 = \frac{1}{N - d} \sum_{n=1}^N (x_n - \mu)^2$$

Onde $d$ é o parâmetro de graus de liberdade ajustado (`ddof`). $d=1$ fornece a variância amostral corrigida (sem viés), enquanto $d=0$ fornece a variância populacional/estimador de máxima verossimilhança.

### Assimetria ($\gamma_1$ - Skewness)
Mede a assimetria da distribuição de probabilidade em relação à sua média. Representa o terceiro momento central padronizado:

$$\gamma_1 = \mathbb{E}\left[ \left( \frac{x - \mu}{\sigma} \right)^3 \right] = \frac{\frac{1}{N} \sum_{n=1}^N (x_n - \mu)^3}{\sigma^3}$$

### Curtose ($\beta_2$ - Kurtosis)
Mede a cauda ou o achatamento da distribuição. Representa o quarto momento central padronizado:

$$\beta_2 = \mathbb{E}\left[ \left( \frac{x - \mu}{\sigma} \right)^4 \right] = \frac{\frac{1}{N} \sum_{n=1}^N (x_n - \mu)^4}{\sigma^4}$$

Para uma distribuição normal pura, $\beta_2 = 3.0$.

---

## 2. Funções de Distribuição de Probabilidade

### Distribuição Binomial
Seja $k$ o número de sucessos obtidos em $n$ ensaios independentes de Bernoulli com probabilidade de sucesso $p$:

$$\text{Bin}(k | n, p) = \binom{n}{k} p^k (1 - p)^{n-k}$$

Onde o coeficiente binomial é definido por:
$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

### Distribuição de Poisson
Descreve a probabilidade de um número $k$ de eventos ocorrerem num intervalo de tempo ou espaço fixo, com uma taxa média $\lambda$:

$$\text{Poi}(k | \lambda) = \frac{e^{-\lambda} \lambda^k}{k!}$$

### Distribuição Normal (Gaussiana)
A densidade de probabilidade da variável contínua $x$ parametrizada pela média $\mu$ e variância $\sigma^2$ é dada por:

$$\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( - \frac{(x - \mu)^2}{2\sigma^2} \right)$$

---

## 3. Estimativa de Máxima Verossimilhança (MLE)
Dada uma amostra independente e identicamente distribuída (i.i.d.) de observações $\mathbf{X} = \{x_1, \dots, x_N\}$, a função de verossimilhança para uma distribuição gaussiana é:

$$p(\mathbf{X} | \mu, \sigma^2) = \prod_{n=1}^N \mathcal{N}(x_n | \mu, \sigma^2)$$

Maximizando o logaritmo da verossimilhança com relação a $\mu$ e $\sigma^2$, obtemos os estimadores de máxima verossimilhança:

$$\mu_{\text{ML}} = \frac{1}{N} \sum_{n=1}^N x_n$$

$$\sigma^2_{\text{ML}} = \frac{1}{N} \sum_{n=1}^N (x_n - \mu_{\text{ML}})^2$$
