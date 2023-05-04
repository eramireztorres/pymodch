from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from pymodch.determ_models.ode_models import OdeModel

class Likelihood(ABC):
    """
    Abstract base class for likelihood functions in Bayesian inference.

    This class represents the likelihood function of a Bayesian model, which evaluates the probability
    of the observed data given the model parameters. It serves as a base class for user-defined likelihood
    functions.

    Methods
    -------
    log_likelihood(data: array, theta: array) -> float
        Evaluate the log-likelihood of the model at the given point in parameter space, given the observed data.

    Notes
    -----
    When implementing a new likelihood function, you should inherit from this class and override the
    `log_likelihood` method to provide your own log-likelihood computation. Make sure to include the
    data and model parameters as input arguments.
    """

    @abstractmethod
    def log_likelihood(self, data: np.array, theta: np.array) -> float:
        """
        Evaluate the log-likelihood of the model at the given point in parameter space, given the observed data.

        This method should be implemented in derived classes to compute the log-likelihood of the model for
        the given data and parameter values.

        Parameters
        ----------
        data : array
            The observed data.
        theta : array
            A point in parameter space.

        Returns
        -------
        float
            The log-likelihood of the model at the specified point in parameter space, given the observed data.
        """
        pass
    
class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood function for Bayesian inference.

    This class represents a Gaussian likelihood function for a Bayesian model, which assumes that the
    observed data follow a Gaussian distribution with a known standard deviation and a mean that depends
    on the model parameters.

    Attributes
    ----------
    known_std_dev : float
        The known standard deviation of the Gaussian distribution.

    Methods
    -------
    log_likelihood(data: array, theta: array) -> float
        Evaluate the log-likelihood of the Gaussian model at the given point in parameter space, given
        the observed data.

    Examples
    --------
    >>> data = np.array([1.5, 2.3, 3.1, 1.7])
    >>> likelihood = GaussianLikelihood(known_std_dev=0.5)
    >>> theta = np.array([2.0])
    >>> likelihood.log_likelihood(data, theta)
    -7.720000000000001

    Notes
    -----
    When using this class in Bayesian inference, ensure that the data and model parameters are
    appropriately scaled and transformed to match the assumptions of a Gaussian distribution.
    """

    def __init__(self, known_std_dev: float):
        """
        Initialize a GaussianLikelihood instance with the specified known standard deviation.

        Parameters
        ----------
        known_std_dev : float
            The known standard deviation of the Gaussian distribution.
        """
        self.known_std_dev = known_std_dev

    def log_likelihood(self, data: np.array, theta: np.array) -> float:
        """
        Evaluate the log-likelihood of the Gaussian model at the given point in parameter space, given
        the observed data.

        Parameters
        ----------
        data : array
            The observed data.
        theta : array
            A point in parameter space, representing the mean of the Gaussian distribution.

        Returns
        -------
        float
            The log-likelihood of the Gaussian model at the specified point in parameter space, given the
            observed data.
        """
        # return -0.5 * np.sum(((data - theta) / self.known_std_dev) ** 2)
    
        return np.sum(stats.norm.logpdf(data, loc=theta, scale=self.known_std_dev))


def get_vector_likelihood(std, data, theta, lh_model):
    """
    Compute the likelihood of the model for each data point.

    Parameters
    ----------
    std : float or array-like
        Standard deviation(s) of the normal distribution(s) used for likelihood calculation.
    data : array-like
        Observed data points.
    theta : array-like
        Model parameters.
    lh_model : LhModel instance
        Likelihood model instance.

    Returns
    -------
    float
        Product of likelihood values for each data point.
    """
    if np.isscalar(std):
        lhs = [stats.norm(loc, std).pdf(data[i]) for i,
               loc in enumerate(lh_model.get_predictions(theta))]
    else:
        lhs = [stats.norm(loc, std[i]).pdf(data[i]) for i,
               loc in enumerate(lh_model.get_predictions(theta))]
    return np.product(np.array(lhs))

class LhModel(Likelihood):
    """
    A class representing a general likelihood model using an ODE model.
    """
    def __init__(self, ode_model: OdeModel, time_vector: np.array):
        """
        Initialize the LhModel class with an ODE model and time vector.

        Parameters
        ----------
        ode_model : OdeModel
            An ODE model instance.
        time_vector : numpy.array
            Time points for the ODE model simulation.
        """
        self.model = ode_model
        self.t = time_vector

    def get_predictions(self, theta):
        """
        Compute model predictions for the given parameters.

        Parameters
        ----------
        theta : array-like
            Model parameters.

        Returns
        -------
        numpy.array
            Model predictions.
        """
        return self.model.simulate_theta(self.t.ravel(), theta)

class LhNormal(LhModel):
    """
    A class representing the likelihood for a normal distribution with an ODE model.
    """
    def __init__(self, ode_model: OdeModel, time_vector: np.array):
        super().__init__(ode_model, time_vector)

    def _get_err_theta(self, theta):
        """
        Get the error term from the model parameters.

        Parameters
        ----------
        theta : array-like
            Model parameters.

        Returns
        -------
        float
            The error term.
        """
        return theta[-1]

    def log_likelihood(self, data, theta):
        """
        Compute the log-likelihood of the model given the data and parameters.

        Parameters
        ----------
        data : array-like
            Observed data points.
        theta : array-like
            Model parameters.

        Returns
        -------
        float
            Log-likelihood of the model given the data and parameters.
        """
        std = self._get_err_theta(theta)
        return get_vector_likelihood(std, data, theta, self)

 
class LhNormalProp(LhNormal):
    def log_likelihood(self, data, theta):
        sigma_std = self._get_err_theta(theta)
        std = sigma_std * data
        return get_vector_likelihood(std, data, theta, self)

class LhBenz(LhModel):
    def __init__(self, ode_model: OdeModel, time_vector: np.array):
        super().__init__(ode_model, time_vector)

    def _get_err_theta(self, theta):
        return theta[-3:]

    def log_likelihood(self, data, theta):
        sigma, Vm, alpha = self._get_err_theta(theta)
        E = np.zeros(len(data))
        for i, Y in enumerate(data):
            if Y < Vm:
                E[i] = Vm**alpha
            else:
                E[i] = Y**alpha
        std = sigma * E
        return get_vector_likelihood(std, data, theta, self)

class LhStudent(LhModel):
    def __init__(self, ode_model: OdeModel, time_vector: np.array):
        super().__init__(ode_model, time_vector)

    def _get_err_theta(self, theta):
        return theta[-2], theta[-1]

    def log_likelihood(self, data, theta):
        k, scale = self._get_err_theta(theta)
        if np.isscalar(scale):
            lhs = [stats.t.logpdf(data[i], k, loc, scale) for i,
                   loc in enumerate(self.get_predictions(theta))]
        else:
            lhs = [stats.t.logpdf(data[i], k, loc, scale[i]) for i,
                   loc in enumerate(self.get_predictions(theta))]
        return np.sum(np.array(lhs))

class LhStudentProp(LhStudent):
    def log_likelihood(self, data, theta):
        k, sig_scale = self._get_err_theta(theta)
        scale = sig_scale * data
        if np.isscalar(scale):
            lhs = [stats.t.logpdf(data[i], k, loc, scale) for i,
                   loc in enumerate(self.get_predictions(theta))]
        else:
            lhs = [stats.t.logpdf(data[i], k, loc, scale[i]) for i,
                   loc in enumerate(self.get_predictions(theta))]
        return np.sum(np.array(lhs))
