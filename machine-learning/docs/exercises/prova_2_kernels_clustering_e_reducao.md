# 📝 Corrective Exam 2: SVM, Clustering, and Dimensionality Reduction (Worked by Hand)

This practical guide is designed as a step-by-step worked exam "by hand," covering margin-based methods, clustering algorithms, and linear projections.

---

## Question 1: Support Vector Machines (SVM), Kernels, and SVR

### 1.1 Derivation of the SVM Dual Formulation
Given the soft-margin maximization with slack variables $\xi_n \ge 0$, the primal Lagrangian is:
$$L_p(\mathbf{w}, b, \boldsymbol{\xi}, \mathbf{a}, \boldsymbol{\mu}) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{n=1}^N \xi_n - \sum_{n=1}^N a_n \{y_n (\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) + b) - 1 + \xi_n\} - \sum_{n=1}^N \mu_n \xi_n$$

Taking derivatives with respect to $\mathbf{w}$, $b$, and $\xi_n$, and setting them to zero:
1.  $\mathbf{w} = \sum_{n=1}^N a_n y_n \boldsymbol{\phi}(\mathbf{x}_n)$
2.  $\sum_{n=1}^N a_n y_n = 0$
3.  $a_n = C - \mu_n \implies 0 \le a_n \le C$ (since $\mu_n \ge 0$)

Substituting these relations back into the Lagrangian yields the max-dual formulation:
$$\max_{\mathbf{a}} \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m y_n y_m K(\mathbf{x}_n, \mathbf{x}_m)$$
Subject to: $0 \le a_n \le C$ and $\sum_{n=1}^N a_n y_n = 0$.

---

### 1.2 Linear SVM by Hand (Classification)
**Worked example**: Given the linearly separable 1D dataset:
*   $x_1 = 1, y_1 = -1$ (Negative class)
*   $x_2 = 2, y_2 = -1$ (Negative class)
*   $x_3 = 4, y_3 = 1$  (Positive class)

Find the maximum-margin separating hyperplane $w x + b = 0$ by hand.
*   **Visual analysis**: The separable region lies between the largest negative value ($x=2$) and the smallest positive value ($x=4$). The optimal cutoff midpoint is $x = 3$.
*   The decision boundary is:
    $$w x + b = 0 \implies x - 3 = 0 \implies w = 1, \quad b = -3$$
*   **Constraint check**:
    *   For $x_1=1$: $y_1(w x_1 + b) = -1(1 \times 1 - 3) = -1(-2) = 2 \ge 1$ (satisfied)
    *   For $x_2=2$: $y_2(w x_2 + b) = -1(1 \times 2 - 3) = -1(-1) = 1 \ge 1$ (satisfied — support vector!)
    *   For $x_3=4$: $y_3(w x_3 + b) = 1(1 \times 4 - 3) = 1(1) = 1 \ge 1$ (satisfied — support vector!)
*   The support vectors are $x_2=2$ and $x_3=4$. The geometric margin is $\gamma = \frac{1}{\|w\|} = 1$.

---

### 1.3 When to Use, and When Not to Use, SVM Variants

| Algorithm | Situation to Use | Situation to Avoid |
| :--- | :--- | :--- |
| **SVM (Classification)** | Well-defined decision margins and high-dimensional data (via the kernel trick). | Very large datasets (quadratic-to-cubic complexity in the number of samples) or very noisy data. |
| **SVR (Regression)** | Complex non-linear relationships in small-to-medium regression problems. | When direct interpretability of model parameters is crucial. |
| **One-Class SVM** | Anomaly/novelty detection in unsupervised settings (only normal-class data available). | When labeled data for both classes is abundantly available (binary classifiers are preferable). |

---

## Question 2: K-Means, DBSCAN, and Spectral Clustering

### 2.1 1D K-Means by Hand
**Worked example**: Given the 1D dataset $X = \{1, 2, 5, 6\}$.
Initialize the centroids at $\mu_1 = 1$, $\mu_2 = 3$. Perform one full iteration of the algorithm.

#### Step 1: Cluster Assignment
Compute the absolute distance from each point to the centroids $\mu_1$ and $\mu_2$:
*   $x_1=1$: $d(1, \mu_1)=0$, $d(1, \mu_2)=2 \implies$ belongs to Cluster 1.
*   $x_2=2$: $d(2, \mu_1)=1$, $d(2, \mu_2)=1 \implies$ arbitrary tie-break $\implies$ Cluster 1.
*   $x_3=5$: $d(5, \mu_1)=4$, $d(5, \mu_2)=2 \implies$ belongs to Cluster 2.
*   $x_4=6$: $d(6, \mu_1)=5$, $d(6, \mu_2)=3 \implies$ belongs to Cluster 2.

*Resulting groups*: $C_1 = \{1, 2\}$, $C_2 = \{5, 6\}$.

#### Step 2: Centroid Update
*   New $\mu_1 = \frac{1 + 2}{2} = 1.5$
*   New $\mu_2 = \frac{5 + 6}{2} = 5.5$

---

### 2.2 1D DBSCAN by Hand
**Worked example**: Given the 1D dataset $X = \{1, 2, 3, 10\}$.
Let $\epsilon = 1.5$ and $\text{MinPts} = 2$. Classify each point.
*   **Point $1$**: Neighbors within distance $\le 1.5$: $N_1 = \{1, 2\}$ (size 2 $\ge$ MinPts) $\implies$ **Core Point**.
*   **Point $2$**: Neighbors: $N_2 = \{1, 2, 3\}$ (size 3 $\ge$ MinPts) $\implies$ **Core Point**.
*   **Point $3$**: Neighbors: $N_3 = \{2, 3\}$ (size 2 $\ge$ MinPts) $\implies$ **Core Point**.
*   **Point $10$**: Neighbors: $N_{10} = \{10\}$ (size 1 $<$ MinPts) $\implies$ **Noise (Outlier)**.

---

### 2.3 Spectral Clustering by Hand
**Worked example**: Consider a 3-node line graph ($1 - 2 - 3$) with unit edge weights ($W_{12}=1, W_{23}=1, W_{13}=0$).
*   **Adjacency matrix $W$**:
    $$W = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
*   **Degree matrix $D$**:
    $$D = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
*   **Laplacian matrix $L = D - W$**:
    $$L = \begin{bmatrix} 1 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{bmatrix}$$
*   The smallest eigenvalue is always $\lambda_1 = 0$, corresponding to the trivial eigenvector $\mathbf{v}_1 = [1, 1, 1]^T$.
*   The second-smallest eigenvalue $\lambda_2$ (the Fiedler eigenvalue) yields the **Fiedler vector** $\mathbf{v}_2$, whose signs define the optimal partition cut of the graph into subgroups disconnected by spectral similarity.

---

### 2.4 When to Use, and When Not to Use, Clustering Methods

| Method | Situation to Use | Situation to Avoid |
| :--- | :--- | :--- |
| **K-Means** | Spherical, well-separated clusters of uniform variance. | Non-convex/non-spherical shapes, disproportionate cluster sizes, or the presence of outliers. |
| **DBSCAN** | Arbitrary geometries or clear spatial noise/anomalies. | Widely varying densities in the data (where a single radius $\epsilon$ fails). |
| **Spectral Clustering** | Clustering on affinity graphs, intricate non-linear manifolds. | Massive problems with memory constraints for eigenvalue decomposition. |

---

## Question 3: Dimensionality Reduction (PCA and SVD)

### 3.1 Theoretical Equivalence Between PCA and SVD
Given the centered matrix $\mathbf{X}_c \in \mathbb{R}^{N \times D}$.
The sample covariance matrix is $\mathbf{S} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c$.

Taking the SVD of $\mathbf{X}_c = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$ and substituting into the covariance:
$$\mathbf{S} = \frac{1}{N - 1} (\mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T)^T (\mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T) = \frac{1}{N-1} \mathbf{V} \boldsymbol{\Sigma}^T \mathbf{U}^T \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$
Since the left singular vectors are orthonormal ($\mathbf{U}^T \mathbf{U} = \mathbf{I}$):
$$\mathbf{S} = \mathbf{V} \left( \frac{\boldsymbol{\Sigma}^2}{N - 1} \right) \mathbf{V}^T$$
This expression is exactly the eigendecomposition of the covariance matrix $\mathbf{S} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^T$. Therefore:
1.  The eigenvectors of $\mathbf{S}$ are the columns of $\mathbf{V}$ (the right singular vectors of $\mathbf{X}_c$).
2.  The exact relationship between eigenvalues and singular values is:
    $$\lambda_i = \frac{\sigma_i^2}{N - 1}$$

---

### 3.2 PCA Projection by Hand
**Worked example**: Given the computed sample covariance matrix:
$$\mathbf{S} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$
Find the eigenvalues and the principal eigenvector to project the data onto 1D, by hand.

#### Step 1: Characteristic Equation
$$\det(\mathbf{S} - \lambda \mathbf{I}) = \det\left(\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix}\right) = 0$$
$$(2-\lambda)^2 - 1 = 0 \implies \lambda^2 - 4\lambda + 3 = 0 \implies (\lambda-3)(\lambda-1) = 0$$
The eigenvalues are $\lambda_1 = 3$ and $\lambda_2 = 1$. The largest eigenvalue (dominant explained variance) is $\lambda_1 = 3$.

#### Step 2: Find the Principal Eigenvector for $\lambda_1 = 3$
$$(\mathbf{S} - 3\mathbf{I})\mathbf{u} = \mathbf{0} \implies \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies -u_1 + u_2 = 0 \implies u_1 = u_2$$
Normalizing the vector to unit norm ($\|\mathbf{u}\| = 1$):
$$\mathbf{u}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
*   **Usage**: Any centered data point $\mathbf{x} = [x, y]^T$ is projected onto the 1D subspace by:
    $$z = \mathbf{u}_1^T \mathbf{x} = \frac{x + y}{\sqrt{2}}$$

---

### 3.3 When to Use, and When Not to Use, PCA and SVD

*   **When to use**: To reduce multicollinearity in regressions, compress images/matrices, or visualize multidimensional data in two-dimensional plots.
*   **When not to use**: When the data's intrinsic structure lies on a complex non-linear manifold (in that case, manifold learning or autoencoders are more suitable), or when direct interpretability of the original variables must be preserved.
