import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import uniform
from pymodch.prior_models.prior_models import UniformPrior, Prior
from pymodch.likelihood_models.likelihood_models import LhNormal
from pymodch.fit.bayes_fit import EmceeModelFitter
from pymodch.determ_models.ode_models import GompertzI

# # Create synthetic data
model = GompertzI()
np.random.seed(42)
t_data = np.linspace(0, 10, 20)
r_true, K_true, V0_true = 0.2, 3000, 500
y_true = model.simulate_theta(t_data, (r_true, K_true, V0_true))
y_err = 50
y_data = y_true + np.random.normal(0, y_err, len(t_data))

#Params limits
lower_bounds = np.array([0, 2000, 480, 40]) 
upper_bounds = np.array([1, 5000, 520, 60])

# lower_bounds = np.array([0, 0, 0, 0]) 
# upper_bounds = np.array([20, 5, 1, 1])

#Define likelihood function with deterministic model plus error 
likelihood = LhNormal(model, t_data, use_square_errors=True)
#Define prior function
prior = UniformPrior(lower_bounds=lower_bounds, upper_bounds=upper_bounds)

# Minimize the negative log-likelihood for a good initial guess
# Just a start point. Any acceptable guess should be enough
def neg_log_likelihood(theta):
    return -likelihood.log_likelihood(y_data, theta)
initial_guess = minimize(neg_log_likelihood, [r_true, K_true, V0_true]).x
initial_guess = np.concatenate((initial_guess, [y_err]))

initial_guess = np.array([0.2, 3000, 500, 50]) 

print(f'initial guess: {initial_guess}')

# Estimate the model parameters with Ensemble Sampler
fitter = EmceeModelFitter(prior, likelihood)
samples = fitter.fit(y_data, initial_guess, n_walkers=50, n_steps=2000)

#%% Get the best-fit parameters
burnin_fraction = 0.5
best_fit_r, best_fit_K, best_fit_V0, _ = np.mean(samples[-np.int32(burnin_fraction*len(samples)):], axis=0)

print(f'true values {[r_true, K_true, V0_true]}')
print(f'best fit {[best_fit_r, best_fit_K, best_fit_V0]}')
print(np.min(samples, axis=0))
print(np.max(samples, axis=0))


#%% Plot the true curve, data points, and the best-fit curve
plt.plot(t_data, y_true, color="k", label="True curve")
plt.errorbar(t_data, y_data, yerr=y_err, fmt="o", markersize=3, capsize=3, label="Data points", alpha=0.6)
y_fit = model.simulate_theta(t_data, (best_fit_r, best_fit_K, best_fit_V0))

# y_fit = gompertzI(t_data, best_fit_A, best_fit_B, best_fit_C)
plt.plot(t_data, y_fit, color="r", linestyle="--", label="Best-fit curve")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.title("GompertzI fit using EmceeModelFitter")
plt.show()

# %%
