import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from pymodch.prior_models.prior_models import UniformPrior
from pymodch.likelihood_models.likelihood_models import Likelihood
from pymodch.fit.bayes_fit import EmceeModelFitter


from scipy.stats import norm

#Define a straight-line likelihood class
class StraightLineLikelihood(Likelihood):
    """
    A class representing the likelihood for a straight line model.
    """
    def __init__(self, x, y, y_err):
        """
        Initialize the StraightLineLikelihood class with data and error.

        Parameters
        ----------
        x : array-like
            Independent variable data points.
        y : array-like
            Dependent variable data points.
        y_err : array-like
            Uncertainties of the dependent variable data points.
        """
        self.x = x
        self.y = y
        self.y_err = y_err

    def log_likelihood(self, data, theta):
        """
        Compute the log-likelihood of the straight line model given the data and parameters.

        Parameters
        ----------
        data : array-like
            Not used in this implementation, but required for consistency with other likelihood models.
        theta : array-like
            Model parameters (slope and intercept).

        Returns
        -------
        float
            Log-likelihood of the straight line model given the data and parameters.
        """
        m, c = theta
        y_model = m * self.x + c
        return np.sum(norm.logpdf(self.y, loc=y_model, scale=1))


# Generate synthetic data
true_m = 2.5
true_b = -1.0
n_points = 100
x = np.linspace(0, 10, n_points)
y_true = true_m * x + true_b
y_err = 0.5 * np.ones_like(x)
y = y_true + y_err * np.random.randn(n_points)

# Create likelihood and prior
likelihood = StraightLineLikelihood(x, y, y_err)
lower_bounds = [-10, -10]
upper_bounds = [10, 10]
prior = UniformPrior(lower_bounds, upper_bounds)

# Instantiate the EmceeModelFitter
n_walkers = 10
n_steps = 200
fitter = EmceeModelFitter(prior, likelihood)

# Fit the data
initial_guess = [1, 1]
samples = fitter.fit(data=y, initial_guess=initial_guess, 
                     n_walkers=n_walkers, n_steps=n_steps)

# Get the best-fit parameters
best_fit_m, best_fit_b = np.median(samples, axis=0)
print(f"Best-fit m: {best_fit_m}, Best-fit b: {best_fit_b}")

# Plot the true line
plt.plot(x, y_true, color="k", label="True line")

# Plot the data points
plt.errorbar(x, y, yerr=y_err, fmt="o", markersize=3, capsize=3, label="Data points", alpha=0.6)

# Plot the best-fit line
y_fit = best_fit_m * x + best_fit_b
plt.plot(x, y_fit, color="r", linestyle="--", label="Best-fit line")

# Customize the plot
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Line fit using EmceeModelFitter")

# Show the plot
plt.show()

