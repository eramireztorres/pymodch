def aic(likelihood, y_data, theta):
    """
    Calculate the Akaike Information Criterion (AIC) for the given likelihood, data, and model parameters.
    
    Parameters
    ----------
    likelihood : Likelihood
        An instance of a likelihood class.
    y_data : array-like
        An array of observed data points.
    theta : array-like
        An array of model parameters.
        
    Returns
    -------
    float
        The AIC value.
    """
    n = len(y_data)
    k = len(theta)
    log_likelihood = likelihood.log_likelihood(y_data, theta)
    return -2 * log_likelihood + 2 * k


def r_squared(likelihood, y_data, theta):
    """
    Calculate the coefficient of determination (R^2) for the given likelihood, data, and model parameters.
    
    Parameters
    ----------
    likelihood : Likelihood
        An instance of a likelihood class.
    y_data : array-like
        An array of observed data points.
    theta : array-like
        An array of model parameters.
        
    Returns
    -------
    float
        The R^2 value.
    """
    model = likelihood.model
    t_data = likelihood.t_data
    y_pred = model.simulate_theta(t_data, theta)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1 - (ss_res / ss_tot)

def adjusted_r_squared(likelihood, y_data, theta):
    """
    Calculate the adjusted coefficient of determination (adjusted R^2) for the given likelihood, data, and model parameters.
    
    Parameters
    ----------
    likelihood : Likelihood
        An instance of a likelihood class.
    y_data : array-like
        An array of observed data points.
    theta : array-like
        An array of model parameters.
        
    Returns
    -------
    float
        The adjusted R^2 value.
    """
    n = len(y_data)
    k = len(theta)
    r2 = r_squared(likelihood, y_data, theta)
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)
