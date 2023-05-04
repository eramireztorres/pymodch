from abc import ABC, abstractmethod
import numpy as np
from scipy import stats

class Prior(ABC):
    """
    Abstract base class for prior probability distributions.

    This class represents the prior probability distribution P(theta) for a Bayesian inference problem,
    where theta denotes the model parameters. It provides methods for evaluating the log prior probability
    and for performing a prior transform.

    Subclasses should implement the `log_prior` and `prior_transform` methods for specific prior distributions.

    Methods
    -------
    log_prior(theta: array) -> float
        Evaluate the log prior probability at the given point in parameter space.
    
    prior_transform(utheta: array) -> array
        Perform the prior transform, which maps a point in the unit hypercube to a point in the parameter space
        according to the prior distribution.
    """

    @abstractmethod
    def log_prior(self, theta: np.array) -> float:
        """
        Evaluate the log prior probability at the given point in parameter space.

        Parameters
        ----------
        theta : array
            A point in parameter space.

        Returns
        -------
        float
            The log prior probability at the specified point.
        """
        pass

    @abstractmethod
    def prior_transform(self, utheta: np.array) -> np.array:
        """
        Perform the prior transform, which maps a point in the unit hypercube to a point in the parameter space
        according to the prior distribution.

        Parameters
        ----------
        utheta : array
            A point in the unit hypercube.

        Returns
        -------
        array
            A point in parameter space corresponding to the specified point in the unit hypercube.
        """
        pass

class GaussianPrior(Prior):
    """
    Gaussian prior probability distribution for Bayesian inference.

    This class represents a Gaussian prior distribution for a Bayesian inference problem,
    where the model parameters follow a multivariate Gaussian distribution with mean vector `mu`
    and covariance matrix sigma^2 * I (I is the identity matrix).

    Inherits from the abstract base class `Prior`.

    Attributes
    ----------
    mu : array
        Mean vector of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Methods
    -------
    log_prior(theta: array) -> float
        Evaluate the log prior probability at the given point in parameter space.
    
    prior_transform(utheta: array) -> array
        Perform the prior transform, which maps a point in the unit hypercube to a point in the parameter space
        according to the prior distribution.
    """

    def __init__(self, mu: np.array, sigma: float):
        """
        Initialize a GaussianPrior instance with the specified mean vector and standard deviation.

        Parameters
        ----------
        mu : array
            Mean vector of the Gaussian distribution.
        sigma : float
            Standard deviation of the Gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def log_prior(self, theta: np.array) -> float:
        """
        Evaluate the log prior probability at the given point in parameter space.

        Parameters
        ----------
        theta : array
            A point in parameter space.

        Returns
        -------
        float
            The log prior probability at the specified point.
        """
        return -0.5 * sum(((theta - self.mu) / self.sigma) ** 2)

    def prior_transform(self, utheta: np.array) -> np.array:
        """
        Perform the prior transform, which maps a point in the unit hypercube to a point in the parameter space
        according to the prior distribution.

        Parameters
        ----------
        utheta : array
            A point in the unit hypercube.

        Returns
        -------
        array
            A point in parameter space corresponding to the specified point in the unit hypercube.
        """
        return self.mu + self.sigma * stats.norm.ppf(utheta)



class UniformPrior(Prior):
    """
    Uniform prior probability distribution for Bayesian inference.

    This class represents a uniform prior distribution for a Bayesian inference problem,
    where the model parameters follow a multivariate uniform distribution within the specified bounds.

    Inherits from the abstract base class `Prior`.

    Attributes
    ----------
    lower_bounds : array
        Lower bounds of the uniform distribution for each parameter.
    upper_bounds : array
        Upper bounds of the uniform distribution for each parameter.

    Methods
    -------
    log_prior(theta: array) -> float
        Evaluate the log prior probability at the given point in parameter space.
    
    prior_transform(utheta: array) -> array
        Perform the prior transform, which maps a point in the unit hypercube to a point in the parameter space
        according to the prior distribution.
    """

    def __init__(self, lower_bounds: np.array, upper_bounds: np.array):
        """
        Initialize a UniformPrior instance with the specified lower and upper bounds.

        Parameters
        ----------
        lower_bounds : array
            Lower bounds of the uniform distribution for each parameter.
        upper_bounds : array
            Upper bounds of the uniform distribution for each parameter.
        """
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def log_prior(self, theta: np.array) -> float:
        """
        Evaluate the log prior probability at the given point in parameter space.

        Parameters
        ----------
        theta : array
            A point in parameter space.

        Returns
        -------
        float
            The log prior probability at the specified point.
        """
        within_bounds = np.all((theta >= self.lower_bounds) & (theta <= self.upper_bounds))
        return 0.0 if within_bounds else -np.inf

    def prior_transform(self, utheta: np.array) -> np.array:
        """
        Perform the prior transform, which maps a point in the unit hypercube to a point in the parameter space
        according to the prior distribution.

        Parameters
        ----------
        utheta : array
            A point in the unit hypercube.

        Returns
        -------
        array
            A point in parameter space corresponding to the specified point in the unit hypercube.
        """
        return self.lower_bounds + utheta * (self.upper_bounds - self.lower_bounds)
