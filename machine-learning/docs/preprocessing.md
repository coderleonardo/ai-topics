# 📊 Pré-processamento e Transformação de Dados

Este documento descreve as formulações matemáticas das técnicas de pré-processamento implementadas no módulo `src/preprocessing.py`.

---

## 1. Min-Max Scaler (MinMaxScaler)
A escala Min-Max projeta linearmente cada atributo para um intervalo especificado $[a, b]$ (por padrão, $[0, 1]$).

Para cada vetor de atributo $\mathbf{x} = [x_1, x_2, \dots, x_N]^T$:

$$x_i^* = \frac{x_i - \min(\mathbf{x})}{\max(\mathbf{x}) - \min(\mathbf{x})} \cdot (b - a) + a$$

Onde:
*   $x_i$ é o valor original do atributo para a amostra $i$.
*   $x_i^*$ é o valor escalado.
*   $\min(\mathbf{x})$ e $\max(\mathbf{x})$ são os valores mínimo e máximo do atributo no conjunto de treinamento.

---

## 2. Standard Scaler (StandardScaler)
A padronização (ou escala Z-score) redimensiona os atributos para que possuam média nula ($\mu = 0$) e variância unitária ($\sigma^2 = 1$).

$$x_i^* = \frac{x_i - \mu}{\sigma}$$

Onde:
*   A média $\mu$ é definida por:
    $$\mu = \frac{1}{N} \sum_{j=1}^{N} x_j$$
*   O desvio padrão $\sigma$ é definido por:
    $$\sigma = \sqrt{\frac{1}{N} \sum_{j=1}^{N} (x_j - \mu)^2}$$

---

## 3. Max-Abs Scaler (MaxAbsScaler)
O escalador absoluto máximo projeta os atributos linearmente de modo que o valor máximo absoluto seja $1$. É adequado para matrizes esparsas, pois preserva o valor zero.

$$x_i^* = \frac{x_i}{\max_{j} |x_j|}$$

---

## 4. Robust Scaler (RobustScaler)
Esta técnica escala os atributos utilizando estatísticas robustas a valores atípicos (outliers). Remove a mediana e escala os dados conforme o Intervalo Interquartílico (IQR).

$$x_i^* = \frac{x_i - \text{mediana}(\mathbf{x})}{\text{IQR}(\mathbf{x})}$$

Onde:
*   $\text{mediana}(\mathbf{x})$ é o valor central ordenado dos dados.
*   $\text{IQR}(\mathbf{x}) = Q_3(\mathbf{x}) - Q_1(\mathbf{x})$, representando a diferença entre o terceiro quartil (75%) e o primeiro quartil (25%).

---

## 5. Janelamento Temporal (make_windows)
Para modelagem autoregressiva de séries temporais $y = [y_1, y_2, \dots, y_T]$, o janelamento mapeia o vetor unidimensional em pares de matrizes de entrada $X$ e vetor de saída $\mathbf{y}$ usando uma janela de tamanho $W$:

$$\mathbf{X}_t = [y_t, y_{t+1}, \dots, y_{t+W-1}] \in \mathbb{R}^W$$
$$y_t^* = y_{t+W} \in \mathbb{R}$$

Dessa forma, o conjunto de dados final possui formato:
*   $X \in \mathbb{R}^{(T-W) \times W}$
*   $\mathbf{y} \in \mathbb{R}^{T-W}$
