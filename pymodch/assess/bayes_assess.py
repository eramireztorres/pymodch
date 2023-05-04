import numpy as np
from functools import partial
import dynesty
from pymodch.prior_models.prior_models import Prior
from pymodch.likelihood_models.likelihood_models import Likelihood


class MarginalLikelihoodEstimator:
    """
    Estimate the marginal likelihood of a Bayesian model using the Nested Sampling algorithm.

    This class estimates the marginal likelihood of a Bayesian model given a prior and likelihood, using the
    Nested Sampling algorithm as implemented in the dynesty package.

    Attributes
    ----------
    prior : Prior
        An instance of a class that inherits from the `Prior` abstract base class, representing the prior
        distribution of the model parameters.
    likelihood : Likelihood
        An instance of a class that inherits from the `Likelihood` abstract base class, representing the
        likelihood function of the model.

    Methods
    -------
    log_likelihood_wrapper(theta: array, data: array) -> float
        Wrapper function for the log-likelihood function of the model, which takes the model parameters
        and data as input and returns the log-likelihood.
    
    estimate(data: array, ndim: int) -> float
        Estimate the marginal likelihood of the Bayesian model using the Nested Sampling algorithm.
    """

    def __init__(self, prior: Prior, likelihood: Likelihood):
        """
        Initialize a MarginalLikelihoodEstimator instance with the specified prior and likelihood.

        Parameters
        ----------
        prior : Prior
            An instance of a class that inherits from the `Prior` abstract base class, representing the prior
            distribution of the model parameters.
        likelihood : Likelihood
            An instance of a class that inherits from the `Likelihood` abstract base class, representing the
            likelihood function of the model.
        """
        self.prior = prior
        self.likelihood = likelihood

    def log_likelihood_wrapper(self, theta: np.array, data: np.array) -> float:
        """
        Wrapper function for the log-likelihood function of the model, which takes the model parameters
        and data as input and returns the log-likelihood.

        Parameters
        ----------
        theta : array
            A point in parameter space.
        data : array
            The observed data.

        Returns
        -------
        float
            The log-likelihood of the model at the specified point in parameter space.
        """
        return self.likelihood.log_likelihood(data, theta)

    def estimate(self, data: np.array, ndim: int) -> float:
        """
        Estimate the marginal likelihood of the Bayesian model using the Nested Sampling algorithm.

        Parameters
        ----------
        data : array
            The observed data.
        ndim : int
            The number of dimensions (parameters) in the model.

        Returns
        -------
        float
            The estimated log marginal likelihood (also known as the log evidence) of the Bayesian model.
        """
        # Instantiate a NestedSampler with a wrapped log_likelihood function
        wrapped_log_likelihood = partial(self.log_likelihood_wrapper, data=data)
        sampler = dynesty.NestedSampler(wrapped_log_likelihood, self.prior.prior_transform, ndim)

        # Run the sampler
        sampler.run_nested()

        # Get results
        results = sampler.results

        # Access the log-evidence (logZ)
        log_evidence = results.logz[-1]

        return log_evidence

def bic(likelihood, y_data, theta):
    """
    Calculate the Bayesian Information Criterion (BIC) for the given likelihood, data, and model parameters.
    
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
        The BIC value.
    """
    n = len(y_data)
    k = len(theta)
    log_likelihood = likelihood.log_likelihood(y_data, theta)
    return -2 * log_likelihood + k * np.log(n)


def dic(likelihood, y_data, theta_samples):
    """
    Calculate the Deviance Information Criterion (DIC) for the given likelihood, data, and model parameters.
    
    Parameters
    ----------
    likelihood : Likelihood
        An instance of a likelihood class.
    y_data : array-like
        An array of observed data points.
    theta_samples : array-like, shape (n_samples, n_parameters)
        An array of sampled model parameters from an MCMC sampler.
        
    Returns
    -------
    float
        The DIC value.
    """
    # Compute the average likelihood for each sample
    avg_likelihood = np.mean([likelihood.log_likelihood(y_data, theta) for theta in theta_samples])
    
    # Compute the deviance for each sample
    deviance_samples = -2 * np.array([likelihood.log_likelihood(y_data, theta) for theta in theta_samples])

    # Compute the average deviance
    avg_deviance = np.mean(deviance_samples)
    
    # Compute the deviance for the average likelihood
    deviance_avg_likelihood = -2 * avg_likelihood
    
    # Compute the effective number of parameters
    p_d = deviance_avg_likelihood - avg_deviance

    # Calculate the DIC
    dic = avg_deviance + 2 * p_d

    return dic
