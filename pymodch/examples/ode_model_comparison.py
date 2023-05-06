import unittest
import numpy as np
from pymodch.prior_models.prior_models import UniformPrior
from pymodch.likelihood_models.likelihood_models import LhNormal, LhNormalProp
from pymodch.determ_models.ode_models import GompertzI, Grm
from pymodch.assess.bayes_assess import MarginalLikelihoodEstimator

class TestModelComparison(unittest.TestCase):
    def test_model_comparison(self):
        np.random.seed(42)  # Set the seed for reproducibility

        # ODE models to compare       
        model_1 = GompertzI()
        model_2 = Grm()
        
        # # Create synthetic data        
        t_data = np.linspace(0, 10, 20)
        r_true, K_true, V0_true = 0.2, 3000, 500
        y_true = model_1.simulate_theta(t_data, (r_true, K_true, V0_true))
        y_err = 50
        y_data = y_true + np.random.normal(0, y_err, len(t_data))
        
        #Params limits
        lower_bounds_1 = np.array([0, 2000, 400, 40]) 
        upper_bounds_1 = np.array([1, 3000, 600, 60])
        
        lower_bounds_2 = np.array([0, 0, 2000, 0, 400, 0]) 
        upper_bounds_2 = np.array([5, 1, 5000, 1, 600, 1])

        # Set up the prior and likelihood models for Model 1
        prior_model1 = UniformPrior(lower_bounds=lower_bounds_1, upper_bounds=upper_bounds_1)
        likelihood_model1 = LhNormal(model_1, t_data)

        # Set up the prior and likelihood models for Model 2
        prior_model2 = UniformPrior(lower_bounds=lower_bounds_2, upper_bounds=upper_bounds_2)
        likelihood_model2 = LhNormalProp(model_2, t_data)

        # Set up the MarginalLikelihoodEstimators for both models
        estimator_model1 = MarginalLikelihoodEstimator(prior_model1, likelihood_model1)
        estimator_model2 = MarginalLikelihoodEstimator(prior_model2, likelihood_model2)

        # Estimate the marginal likelihood for both models
        ndim_1 = len(lower_bounds_1)
        ndim_2 = len(lower_bounds_2)
        log_evidence_model1 = estimator_model1.estimate(y_data, ndim_1)
        log_evidence_model2 = estimator_model2.estimate(y_data, ndim_2)

        # Check if the estimates are reasonable
        self.assertIsNotNone(log_evidence_model1)
        self.assertIsNotNone(log_evidence_model2)
        self.assertTrue(np.isfinite(log_evidence_model1))
        self.assertTrue(np.isfinite(log_evidence_model2))

        # Compare the models (Bayes factor)
        log_bayes_factor = log_evidence_model1 - log_evidence_model2
        print(f'Log Bayes factor: {log_bayes_factor}')
        self.assertTrue(log_bayes_factor > 0)  # Model 1 should have a higher evidence


if __name__ == '__main__':
    unittest.main()
    