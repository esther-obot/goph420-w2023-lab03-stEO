import numpy as np


def multi_regress(y, Z):
    """Perform multiple linear regression.
    
    Parameters
    ----------
    y : array_like, shape = (n,) or (n,1)
        The vector of dependent variable data
    Z : array_like, shape = (n,m)
        The matrix of independent variable data
    
    Returns
    -------
    numpy.ndarray, shape = (m,) or (m,1)
        The vector of model coefficients
    numpy.ndarray, shape = (n,) or (n,1)
        The vector of residuals
    float
        The coefficient of determination, r^2
    """
    y = np.array(y)
    Z = np.array(Z)
    a = np.linalg.solve(Z.T@Z, Z.T@y)
    f= Z@a
    e= y-f
    ebar =y - np.mean(y)
    phi = e.T@e
    phi_bar=ebar.T@ebar
    rsq= (phi_bar-phi)/phi_bar
    return a,e,rsq

