# 📝 Corrective Exam 1: Statistical Theory, Sampling and Linear Models (Worked by Hand)

This practical guide is designed as a step-by-step worked exam "by hand," serving as study material to reinforce the arithmetic, concepts, and modeling decisions involved.

---

## Question 1: Distributions, Maximum Likelihood, and Sampling Methods

### 1.1 Derivation of the Variance Estimator (MLE)
Let $\{x_1, \dots, x_N\}$ be an i.i.d. sample of $x_n \sim \mathcal{N}(\mu, \sigma^2)$.

The log-likelihood function is:
$$\ln L(\mu, \sigma^2) = -\frac{N}{2} \ln(2\pi) - \frac{N}{2} \ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{n=1}^N (x_n - \mu)^2$$

To find the MLE estimator of the variance $\sigma^2_{\text{ML}}$, let $\theta = \sigma^2$, differentiate with respect to $\theta$, and set to zero:
$$\frac{\partial \ln L}{\partial \theta} = -\frac{N}{2\theta} + \frac{1}{2\theta^2} \sum_{n=1}^N (x_n - \mu)^2 = 0 \implies \sigma^2_{\text{ML}} = \frac{1}{N} \sum_{n=1}^N (x_n - \mu)^2$$

**Bias proof**: The expected value of the mean squared deviation is:
$$\mathbb{E}[\sigma^2_{\text{ML}}] = \frac{N-1}{N}\sigma^2$$
Since $\mathbb{E}[\sigma^2_{\text{ML}}] \neq \sigma^2$, the estimator is **biased** downward. Bessel's correction divides by $N-1$ instead of $N$ to remove this bias.

---

### 1.2 Sampling and Validation Methods (Holdout, K-Fold, and Bootstrap)

#### A) K-Fold Partition Splitting by Hand
**Worked example**: Given the dataset $\mathcal{D} = \{A, B, C, D, E, F\}$ ($N=6$) and $K=3$ folds, how are the partitions generated?
*   Size of each fold: $N / K = 6 / 3 = 2$ samples per fold.
*   **Fold 1**: Test = $\{A, B\}$; Train = $\{C, D, E, F\}$
*   **Fold 2**: Test = $\{C, D\}$; Train = $\{A, B, E, F\}$
*   **Fold 3**: Test = $\{E, F\}$; Train = $\{A, B, C, D\}$

#### B) Out-of-Bag (OOB) Probability in Bootstrap, by Hand
**Worked example**: In Bootstrap, we sample $N$ elements with replacement. For $N=3$ elements ($A, B, C$), what is the exact probability that element $A$ is left out of the sample (OOB)?
*   The probability that $A$ is not chosen in a single draw is $2/3$.
*   Over $N=3$ independent draws with replacement:
    $$P(A \text{ is OOB}) = \left(\frac{2}{3}\right)^3 = \frac{8}{27} \approx 0.296 \quad (29.6\%)$$
*   *Note*: As $N \to \infty$, this probability converges to $1/e \approx 36.8\%$.

---

### 1.3 When to Use, and When Not to Use, Each Sampling Method

| Method | Situation to Use | Situation to Avoid |
| :--- | :--- | :--- |
| **Holdout (Train/Test)** | Very large datasets (where running K-Fold is computationally infeasible). | Very small datasets (the test set has high variance and training loses valuable data). |
| **K-Fold Cross-Validation** | Robust hyperparameter evaluation on medium-sized data. | Time series or sequential data (where shuffling breaks temporal dependence). |
| **Bootstrap (OOB)** | Small datasets and ensemble models (like Random Forests) for estimating uncertainty. | When the computational cost of retraining the model hundreds of times is prohibitive. |

---

## Question 2: Simple Linear Regression, ANOVA, and the t-Test by Hand

### 2.1 Step-by-Step Calculations with Small Numbers
Given the sample with $N=4$ points:
$$\mathcal{D} = \{(1, 2), (2, 4), (3, 5), (4, 7)\}$$

#### Step 1: Means
$$\bar{x} = \frac{1+2+3+4}{4} = 2.5, \quad \bar{y} = \frac{2+4+5+7}{4} = 4.5$$

#### Step 2: $S_{xx}$ and $S_{xy}$
*   $S_{xx} = \sum (x_i - \bar{x})^2 = (-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5.0$
*   $S_{xy} = \sum (x_i - \bar{x})(y_i - \bar{y}) = (-1.5)(-2.5) + (-0.5)(-0.5) + (0.5)(0.5) + (1.5)(2.5) = 3.75 + 0.25 + 0.25 + 3.75 = 8.0$

#### Step 3: Parameters $\beta_1$ and $\beta_0$
*   $\beta_1 = \frac{S_{xy}}{S_{xx}} = \frac{8.0}{5.0} = 1.6$
*   $\beta_0 = \bar{y} - \beta_1\bar{x} = 4.5 - 1.6(2.5) = 0.5$
*   *Fitted line*: $\hat{y} = 0.5 + 1.6x$

#### Step 4: Fitted Values and Residuals
*   $\hat{y} = [2.1, 3.7, 5.3, 6.9]$
*   $e = y - \hat{y} = [2-2.1, 4-3.7, 5-5.3, 7-6.9] = [-0.1, 0.3, -0.3, 0.1]$

#### Step 5: ANOVA and $R^2$
*   **SSE** (Sum of Squared Errors): $\sum e_i^2 = (-0.1)^2 + 0.3^2 + (-0.3)^2 + 0.1^2 = 0.01 + 0.09 + 0.09 + 0.01 = 0.20$
*   **SST** (Total Sum of Squares): $\sum (y_i - \bar{y})^2 = (-2.5)^2 + (-0.5)^2 + 0.5^2 + 2.5^2 = 6.25 + 0.25 + 0.25 + 6.25 = 13.00$
*   **SSR** (Regression Sum of Squares): $\text{SST} - \text{SSE} = 13.00 - 0.20 = 12.80$
*   **$R^2$**: $\frac{\text{SSR}}{\text{SST}} = \frac{12.80}{13.00} \approx 0.9846 \quad (98.46\%)$

#### Step 6: Variance and Standard Error
*   $\hat{\sigma}^2 = \frac{\text{SSE}}{N - 2} = \frac{0.20}{2} = 0.10$
*   $\text{SE}(\beta_1) = \sqrt{\frac{\hat{\sigma}^2}{S_{xx}}} = \sqrt{\frac{0.10}{5.0}} = \sqrt{0.02} \approx 0.1414$

#### Step 7: Significance $t$-Statistic
*   $t = \frac{\beta_1}{\text{SE}(\beta_1)} = \frac{1.6}{0.1414} \approx 11.31$
*   Since $|t| = 11.31 > t_{\text{crit}} = 4.303$ (for 2 degrees of freedom at $\alpha=0.05$), we reject $H_0$. The model is statistically useful.

---

### 2.2 When to Use, and When Not to Use, Linear Regression

*   **When to use**: When the relationship between the response variable and the predictors is approximately linear and we want to interpret the magnitude of each feature's impact (statistical inference).
*   **When not to use**: When the residuals exhibit heteroscedastic behavior (non-constant variance) or pronounced non-linearity (e.g., exponential growth).

---

## Question 3: Regularization (Ridge) and KNN by Hand

### 3.1 1D Ridge Regression by Hand
Without an intercept, the Ridge coefficient $\beta_{\text{Ridge}}$ for one-dimensional data with penalty $\lambda$ solves the minimization of:
$$E(w) = \frac{1}{2} \sum_{i=1}^N (y_i - w x_i)^2 + \frac{\lambda}{2} w^2 \implies w_{\text{Ridge}} = \frac{\sum_{i=1}^N x_i y_i}{\sum_{i=1}^N x_i^2 + \lambda}$$

**Worked example**: Given the points $(1, 2)$ and $(2, 3)$, compute $w_{\text{Ridge}}$ with $\lambda = 1.0$:
*   $\sum x_i y_i = (1 \times 2) + (2 \times 3) = 2 + 6 = 8$
*   $\sum x_i^2 = 1^2 + 2^2 = 5$
*   $$w_{\text{Ridge}} = \frac{8}{5 + 1.0} = \frac{8}{6} \approx 1.33$$
*   *Note (OLS comparison)*: Without regularization ($\lambda=0$), we would have $w_{\text{OLS}} = 8/5 = 1.60$. The Ridge penalty shrank the coefficient to $1.33$.

---

### 3.2 k-Nearest Neighbors (KNN) by Hand
**Worked example**: Given the training set below:
*   $\mathbf{x}_1 = (1, 1), y_1 = 0$
*   $\mathbf{x}_2 = (1, 2), y_2 = 0$
*   $\mathbf{x}_3 = (3, 3), y_3 = 1$
*   $\mathbf{x}_4 = (4, 3), y_4 = 1$

Classify the query point $\mathbf{x}^* = (2, 2)$ using $K=3$ neighbors with **Euclidean ($L_2$) distance**.
*   $d_2(\mathbf{x}^*, \mathbf{x}_1) = \sqrt{(2-1)^2 + (2-1)^2} = \sqrt{2} \approx 1.414$ (Class 0)
*   $d_2(\mathbf{x}^*, \mathbf{x}_2) = \sqrt{(2-1)^2 + (2-2)^2} = 1.000$ (Class 0)
*   $d_2(\mathbf{x}^*, \mathbf{x}_3) = \sqrt{(2-3)^2 + (2-3)^2} = \sqrt{2} \approx 1.414$ (Class 1)
*   $d_2(\mathbf{x}^*, \mathbf{x}_4) = \sqrt{(2-4)^2 + (2-3)^2} = \sqrt{5} \approx 2.236$ (Class 1)

The 3 nearest neighbors are $\mathbf{x}_2$ ($d=1.0$), $\mathbf{x}_1$ ($d\approx 1.414$) and $\mathbf{x}_3$ ($d\approx 1.414$).
*   Votes: Class 0 (2 votes), Class 1 (1 vote).
*   **Result**: Predicted class for $\mathbf{x}^*$ is **0**.

---

### 3.3 When to Use, and When Not to Use, Ridge, Lasso, and KNN

| Algorithm | Situation to Use | Situation to Avoid |
| :--- | :--- | :--- |
| **Ridge (L2)** | Strong multicollinearity (highly correlated features) and many features. | When feature selection and zeroing out irrelevant coefficients is essential. |
| **Lasso (L1)** | Automatic feature selection (sparsity) in problems with many irrelevant variables. | When we have highly correlated variables (Lasso tends to pick one at random and drop the rest). |
| **KNN** | Local, non-linear decision patterns in low dimensionality. | High dimensionality (Curse of Dimensionality) or massive datasets (slow at prediction time). |
