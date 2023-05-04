# PyModCh

PyModCh is a Python package for **Bayesian model comparison** and **parameter estimation** using ensemble Markov chain Monte Carlo (MCMC) techniques. This package is designed to be user-friendly, flexible, and efficient for fitting and comparing different models.

## Features

- Provides a suite of **likelihood**, **prior**, and **deterministic models** to perform Bayesian parameter estimation and model comparison.
- Implements the `EmceeModelFitter`, which utilizes the [emcee](https://emcee.readthedocs.io/en/stable/) package for ensemble MCMC sampling.
- Supports the estimation of **marginal likelihoods** and the computation of **Bayes factors** for model comparison.
- Includes several example scripts demonstrating how to use the package for fitting different models to synthetic data.

## Installation

To install PyModCh, simply clone this repository and run:

```bash
pip install .
```

## Examples

Three examples are provided to demonstrate how to use PyModCh for Bayesian parameter estimation and model comparison:

1. `line_fit.py` illustrates how to fit a straight line model to synthetic data using the `EmceeModelFitter`.
2. `gompertz_fit.py` demonstrates how to fit a Gompertz model to synthetic data using the `EmceeModelFitter`.
3. `simple_model_comparison.py` provides a unit test to compare two Gaussian models based on their marginal likelihoods.

To run the examples, navigate to the examples directory and execute the corresponding Python scripts:

```bash
python line_fit.py
python gompertz_fit.py
python simple_model_comparison.py
```

## Usage

To use PyModCh in your own projects, simply import the required modules and classes:

```bash
from pymodch.prior_models.prior_models import UniformPrior
from pymodch.likelihood_models.likelihood_models import StraightLineLikelihood
from pymodch.fit.bayes_fit import EmceeModelFitter
```

Then, define your likelihood and prior models, instantiate an EmceeModelFitter, and fit your data.

For more detailed usage instructions, please refer to the provided examples.

## Contributing

We welcome contributions to PyModCh. If you would like to contribute, please submit a pull request with your proposed changes.

## License

PyModCh is released under the MIT license.
