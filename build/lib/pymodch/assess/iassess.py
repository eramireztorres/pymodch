from abc import ABC, abstractmethod
from pymodch.likelihood_models.Ilikelihood_models import ILikelihoodModel


class IModelAssessor(ABC):
    """
    Interface for one likelihood model evaluator
    Declares the operations that model assessors
    must implement.
    """    
    def __init__(self, model: ILikelihoodModel):
        self.model = model

    @abstractmethod
    def assess(self, *args, **kwargs):
        """
        To be implemented
        Returns the value(s) of model assessment
        Raises:
            NotImplementedError
        """
        raise NotImplementedError("This needs to be implemented")
    
    
class I2ModelAssessor(ABC):
    """
    Interface for two likelihood models' evaluator
    Declares the operations that two model assessors
    (like Bayes factors' estimators)
    must implement.
    """  
      
    def __init__(self, model_1: ILikelihoodModel, model_2: ILikelihoodModel):
        self.model_1 = model_1
        self.model_2 = model_2

    @abstractmethod
    def assess(self, *args, **kwargs):
        """
        To be implemented
        Returns the value(s) of models assessment
        Raises:
            NotImplementedError
        """
        raise NotImplementedError("This needs to be implemented")
    