import unittest
import numpy as np
from pymodch.prior_models.prior_models import GaussianPrior
from pymodch.likelihood_models.likelihood_models import GaussianLikelihood
from pymodch.assess.bayes_assess import MarginalLikelihoodEstimator

class TestModelComparison(unittest.TestCase):
    def test_model_comparison(self):
        np.random.seed(42)  # Set the seed for reproducibility

        # Generate synthetic data
        true_mean = 0.0
        known_std_dev = 1.0
        n_data_points = 100
        data = np.random.normal(loc=true_mean, scale=known_std_dev, size=n_data_points)

        # Set up the prior and likelihood models for Model 1
        prior_model1 = GaussianPrior(mu=0, sigma=1)
        likelihood_model1 = GaussianLikelihood(known_std_dev=known_std_dev)

        # Set up the prior and likelihood models for Model 2
        prior_model2 = GaussianPrior(mu=2, sigma=3)
        likelihood_model2 = GaussianLikelihood(known_std_dev=known_std_dev)

        # Set up the MarginalLikelihoodEstimators for both models
        estimator_model1 = MarginalLikelihoodEstimator(prior_model1, likelihood_model1)
        estimator_model2 = MarginalLikelihoodEstimator(prior_model2, likelihood_model2)

        # Estimate the marginal likelihood for both models
        ndim = 1
        log_evidence_model1 = estimator_model1.estimate(data, ndim)
        log_evidence_model2 = estimator_model2.estimate(data, ndim)

        # Check if the estimates are reasonable
        self.assertIsNotNone(log_evidence_model1)
        self.assertIsNotNone(log_evidence_model2)
        self.assertTrue(np.isfinite(log_evidence_model1))
        self.assertTrue(np.isfinite(log_evidence_model2))

        # Compare the models (Bayes factor)
        log_bayes_factor = log_evidence_model1 - log_evidence_model2
        print(f'ML model 1: {np.exp(log_evidence_model1)}')
        print(f'ML model 2: {np.exp(log_evidence_model2)}')
        print(f'Bayes factor: {np.exp(log_bayes_factor)}')
        self.assertTrue(log_bayes_factor > 0)  # Model 1 should have a higher evidence


if __name__ == '__main__':
    unittest.main()
    