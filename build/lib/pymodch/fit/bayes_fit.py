import emcee
import numpy as np
from typing import Callable


class EmceeModelFitter:
    """
    This class provides a method for estimating model parameters using emcee, a Markov Chain Monte Carlo (MCMC) ensemble sampler.

    Attributes:
    likelihood (Likelihood): An instance of a class that inherits from the Likelihood base class, which
    computes the log likelihood for the given data and model parameters.
    prior (Prior): An instance of a class that inherits from the Prior base class, which computes the log prior
    probability for the model parameters.
    n_walkers (int): The number of walkers in the MCMC ensemble. Default is 50.
    n_steps (int): The total number of steps for each walker in the MCMC chain. Default is 5000.

    Methods:
    fit(data, initial_guess, n_burnin=1000): Estimates the model parameters using the emcee sampler. This function
    takes the observed data, an initial guess for the model parameters,
    and the number of burn-in steps. It returns the MCMC samples for the
    model parameters after discarding the burn-in steps and thinning the chain.

    Example:

    makefile

    likelihood = GaussianLikelihood(known_std_dev)
    prior = GaussianPrior(mu, sigma)
    fitter = EmceeModelFitter(likelihood, prior)

    # Assuming some data and initial_guess for the parameters
    data = ...
    initial_guess = ...

    samples = fitter.fit(data, initial_guess)

    """
    def __init__(self, prior: Callable, likelihood: Callable):
        self.prior = prior
        self.likelihood = likelihood

    def _log_probability(self, theta, data):
        log_prior = self.prior.log_prior(theta)
        if np.isfinite(log_prior):
            return log_prior + self.likelihood.log_likelihood(data, theta)
        return -np.inf

    def fit(self, data, initial_guess, n_walkers=50, n_steps=2000):
        ndim = len(initial_guess)
        initial_positions = initial_guess + 1e-4 * np.random.randn(n_walkers, ndim)

        sampler = emcee.EnsembleSampler(n_walkers, ndim, self._log_probability, args=(data,))
        sampler.run_mcmc(initial_positions, n_steps, progress=True)

        samples = sampler.get_chain(discard=100, thin=15, flat=True)
        return samples
    