# Package entry point for machine learning from scratch.

from .preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
    make_windows
)

from .knn import (
    KNNClassifier,
    KNNRegressor
)

from .distributions import (
    mean_custom,
    variance_custom,
    std_dev_custom,
    skewness_custom,
    kurtosis_custom,
    binomial_pmf,
    poisson_pmf,
    normal_pdf,
    fit_normal_mle
)

from .linear_models import (
    CustomLeastSquares,
    CustomLinearRegression,
    CustomRidgeRegression,
    CustomLassoRegression,
    teste_utilidade_regressao,
    estimativa_variancia,
    analise_qualidade_regressao
)

from .svm import (
    CustomSVM_Dual,
    CustomSVR_Dual,
    CustomSVM_Multiclass,
    CustomOneClassSVM,
    rbf_kernel
)

from .clustering import (
    CustomKMeans,
    CustomDBSCAN,
    CustomSpectralClustering,
    silhouette_values
)

from .dimensionality_reduction import (
    CustomPCA,
    CustomSVD,
    pca_manual,
    svd_estavel_truncada
)

from .metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from .sampling import (
    train_test_split,
    KFold,
    kfold_cross_validation,
    bootstrap_validation
)
