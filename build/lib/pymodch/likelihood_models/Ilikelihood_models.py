from abc import ABC, abstractmethod
from numpy import array


class ILikelihoodModel(ABC):
    """
    Interface for likelihood models f(data|theta)
    Declares the operations that likelihood models
    must implement.
    """

    @abstractmethod
    def get_log_likelihood(self, data: array, theta: array, *args, **kwargs) -> float:
        """
        To be implemented
        Returns the log likelihood given data and parameter vector 
        Args:
            data (array): dataset
            theta (array): parameter vector

        Raises:
            NotImplementedError

        Returns:
            float: log likelihood
        """
        raise NotImplementedError("This needs to be implemented")