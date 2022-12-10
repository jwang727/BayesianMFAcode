import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

from preprocessingfun import sigmastd

def gaussianmodel(mu, Sigma, X, datavector):
    """
    Function for Gaussian model
    """

    posteriormeangauss = mu + Sigma @ np.transpose(X) @ np.linalg.inv(
        X @ Sigma @ np.transpose(X) + (sigmastd ** 2) * np.eye(len(datavector))) @ (datavector - X @ mu)
    posteriorcovariancegauss = Sigma - Sigma @ np.transpose(X) @ np.linalg.inv(
        X @ Sigma @ np.transpose(X) + (sigmastd ** 2) * np.eye(len(datavector))) @ X @ Sigma

    return posteriormeangauss, posteriorcovariancegauss


def Ridgemodel(X, y, alpha):
    """
    Function for Ridge regression, imported from scipy
    """

    regr = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    regr.fit(X, y)

    return regr.coef_


def MLPmodel(X, y):
    """
    Function for Multilayer Perceptron, imported from scipy
    """

    regr = MLPRegressor(random_state=123456, max_iter=1000).fit(X, y)
    mlppredict = regr.predict(np.identity(X.shape[1]))

    return mlppredict
