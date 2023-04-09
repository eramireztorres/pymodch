import numpy as np
from scipy import stats
from pymodch.likelihood_models.Ilikelihood_models import ILikelihoodModel
from pymodch.determ_models.ode_models import OdeModel

class LhModel:
    def __init__(self, ode_model: OdeModel, time_vector: np.array):
        self.model = ode_model
        self.t = time_vector

def get_vector_likelihood(std, data, theta, lh_model):
    if np.isscalar(std):
        lhs = [stats.norm(loc, std).logpdf(data[i]) for i,
               loc in enumerate(lh_model.get_predictions(theta))]
    else:
        lhs = [stats.norm(loc, std[i]).logpdf(data[i]) for i,
               loc in enumerate(lh_model.get_predictions(theta))]
    return np.sum(np.array(lhs))


class LhNormal(ILikelihoodModel, LhModel): 
    def _get_err_theta(self, theta):
        return theta[-1]
    
    def get_residual_correct_vector(self, *args, **kwargs):
        return 1

    def get_predictions(self, theta): 
        "before call model simulate theta in LhNormal"
        return self.model.simulate_theta(self.t.ravel(), theta)

    def get_log_likelihood(self, data, theta):
        std = self._get_err_theta(theta) 
        return get_vector_likelihood(std, data, theta, self)
   

class LhNormalProp(ILikelihoodModel, LhModel): 
    def _get_err_theta(self, theta):
        return theta[-1]
    
    def get_residual_correct_vector(self, data, *args, **kwargs):
        return data

    def _get_predictions(self, theta): 
        "before call model simulate theta in LhNormal"
        return self.model.simulate_theta(self.t.ravel(), theta)
    
    def get_log_likelihood(self, data, theta):
        sigma_std = self._get_err_theta(theta)
        std = sigma_std*data
        return get_vector_likelihood(std, data, theta, self)
    

class LhBenz(ILikelihoodModel, LhModel):   
    def _get_err_theta(self, theta):
        return theta[-3:]
 
    def get_residual_correct_vector(self, data, theta, *args, **kwargs):
        y = data
        E = np.zeros(len(y))
        alpha = theta[-1]
        Vm = theta[-2]
        for i, Y in enumerate(y):
            if Y < Vm:
                E[i] = Vm**alpha
            else:
                E[i] = Y**alpha
        return E

    def _get_predictions(self, theta): 
        "before call model simulate theta in LhNormal"
        return self.model.simulate_theta(self.t.ravel(), theta)
    
    def get_log_likelihood(self, data, theta):
        sigma, Vm, alpha = self._get_err_theta(theta)
        E = np.zeros(len(data))
        for i, Y in enumerate(data):
            if Y < Vm:
                E[i] = Vm**alpha
            else:
                E[i] = Y**alpha        
        std = sigma*E
        return get_vector_likelihood(std, data, theta, self)
    
class LhStudent(ILikelihoodModel, LhModel): 
    def _get_err_theta(self, theta):
        return theta[-2], theta[-1]
    
    def get_residual_correct_vector(self, *args, **kwargs):
        return 1

    def get_predictions(self, theta): 
        "before call model simulate theta in LhNormal"
        return self.model.simulate_theta(self.t.ravel(), theta)

    def get_log_likelihood(self, data, theta):
        k, scale = self._get_err_theta(theta)       
        if np.isscalar(scale):
            lhs = [stats.t.logpdf(data[i], k, loc, scale) for i,
                   loc in enumerate(self.get_predictions(theta))]
        else:
            lhs = [stats.t.logpdf(data[i], k, loc, scale[i]) for i,
                   loc in enumerate(self.get_predictions(theta))]
        return np.sum(np.array(lhs))
    

class LhStudentProp(ILikelihoodModel, LhModel): 
    def _get_err_theta(self, theta):
        return theta[-2], theta[-1]
    
    def get_residual_correct_vector(self, *args, **kwargs):
        return 1

    def get_predictions(self, theta): 
        "before call model simulate theta in LhNormal"
        return self.model.simulate_theta(self.t.ravel(), theta)

    def get_log_likelihood(self, data, theta):
        k, sig_scale = self._get_err_theta(theta) 
        scale = sig_scale*data      
        if np.isscalar(scale):
            lhs = [stats.t.logpdf(data[i], k, loc, scale) for i,
                   loc in enumerate(self.get_predictions(theta))]
        else:
            lhs = [stats.t.logpdf(data[i], k, loc, scale[i]) for i,
                   loc in enumerate(self.get_predictions(theta))]
        return np.sum(np.array(lhs))
 