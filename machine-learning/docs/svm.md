# 🛡️ Máquinas de Vetores de Suporte (Support Vector Machines - SVM)

Este documento detalha as formulações duais clássicas do Bishop (PRML) para Máquinas de Vetores de Suporte (SVM), Regressão por Vetores de Suporte (SVR) e SVM de Classe Única (One-Class SVM) implementadas em `src/svm.py`.

---

## 1. SVM Dual para Classificação Binária (Margem Suave)
Dado um conjunto de dados $\{(\mathbf{x}_n, y_n)\}_{n=1}^N$ com $y_n \in \{-1, +1\}$, a formulação dual busca maximizar os multiplicadores de Lagrange $\mathbf{a} = [a_1, \dots, a_N]^T$:

$$\max_{\mathbf{a}} \tilde{L}(\mathbf{a}) = \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m y_n y_m k(\mathbf{x}_n, \mathbf{x}_m)$$

Sujeito às restrições lineares e de caixa:

$$0 \le a_n \le C, \quad n=1,\dots,N$$
$$\sum_{n=1}^N a_n y_n = 0$$

Onde $C > 0$ regula a tolerância aos erros de classificação na margem.
A função de decisão para um novo ponto $\mathbf{x}$ é expressa como:

$$y(\mathbf{x}) = \text{sign} \left( \sum_{n \in S} a_n y_n k(\mathbf{x}, \mathbf{x}_n) + b \right)$$

Onde $S$ é o conjunto de vetores de suporte ($a_n > 0$). O intercepto (bias) $b$ é obtido a partir de qualquer ponto $m$ cuja restrição seja estrita ($0 < a_m < C$):

$$b = y_m - \sum_{n \in S} a_n y_n k(\mathbf{x}_m, \mathbf{x}_n)$$

---

## 2. Regressão por Vetores de Suporte (SVR)
Para estimar funções contínuas usando uma função de perda robusta a ruídos $\epsilon$-insensitiva, a formulação dual busca maximizar os multiplicadores $\mathbf{a}$ e $\mathbf{a}^*$:

$$\max_{\mathbf{a}, \mathbf{a}^*} \tilde{L}(\mathbf{a}, \mathbf{a}^*) = - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N (a_n - a_n^*) (a_m - a_m^*) k(\mathbf{x}_n, \mathbf{x}_m) - \epsilon \sum_{n=1}^N (a_n + a_n^*) + \sum_{n=1}^N (a_n - a_n^*) y_n$$

Sujeito a:

$$0 \le a_n, a_n^* \le C, \quad n=1,\dots,N$$
$$\sum_{n=1}^N (a_n - a_n^*) = 0$$

A previsão para um ponto $\mathbf{x}$ é:

$$y(\mathbf{x}) = \sum_{n=1}^N (a_n - a_n^*) k(\mathbf{x}, \mathbf{x}_n) + b$$

O bias $b$ é calculado para pontos com $0 < a_m < C$:
$$b = y_m - \epsilon - \sum_{n=1}^N (a_n - a_n^*) k(\mathbf{x}_m, \mathbf{x}_n)$$

---

## 3. SVM de Classe Única (One-Class SVM)
O objetivo é delinear uma fronteira de decisão compacta para a detecção de novidades e anomalias. Minimizamos no dual (formulação de Schölkopf):

$$\min_{\mathbf{a}} \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m k(\mathbf{x}_n, \mathbf{x}_m)$$

Sujeito a:

$$0 \le a_n \le \frac{1}{\nu N}, \quad n=1,\dots,N$$
$$\sum_{n=1}^N a_n = 1$$

Onde $\nu \in (0, 1]$ estabelece um limite superior para a fração de anomalias e um limite inferior para a fração de vetores de suporte.
A função de decisão é:

$$f(\mathbf{x}) = \text{sign} \left( \sum_{n=1}^N a_n k(\mathbf{x}, \mathbf{x}_n) - \rho \right)$$

O limiar $\rho$ é calculado a partir de um vetor de suporte $m$ no interior da fronteira ($0 < a_m < \frac{1}{\nu N}$):

$$\rho = \sum_{n=1}^N a_n k(\mathbf{x}_m, \mathbf{x}_n)$$

---

## 4. Kernels Implementados
O kernel padrão é o Kernel RBF (Função de Base Radial):

$$k(\mathbf{x}, \mathbf{z}) = \exp \left( -\gamma \|\mathbf{x} - \mathbf{z}\|^2 \right)$$
Onde $\gamma > 0$ controla o raio de influência das amostras de treinamento.
