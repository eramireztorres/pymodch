from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import odeint

class OdeModel(ABC):
    """
    An abstract base class for defining ordinary differential equation (ODE) models.
    """
    @staticmethod
    @abstractmethod
    def simulate_theta(t, theta, *args, **kwargs):
        """
        Simulate the ODE model using the given time points and parameters.
        
        Parameters
        ----------
        t : array-like
            Time points at which to evaluate the model.
        theta : array-like
            Model parameters.
        *args : tuple
            Additional arguments to pass to the simulation function.
        **kwargs : dict
            Additional keyword arguments to pass to the simulation function.

        Returns
        -------
        array-like
            Simulated values of the ODE model at the given time points.
        """
        pass

def first_derivative_estimator(C):
    """
    Compute the first derivative of the input array C.

    Parameters
    ----------
    C : array-like
        Input array.

    Returns
    -------
    list
        List of first derivative values.
    """
    Cz = np.hstack(([0], C))
    Diff1 = [Cz[i]-Cz[i-1] for i in range(1, len(Cz))]
    return Diff1

def second_derivative_estimator(C):
    """
    Compute the second derivative of the input array C.

    Parameters
    ----------
    C : array-like
        Input array.

    Returns
    -------
    list
        List of second derivative values.
    """
    Cz = np.hstack(([0, 0], C))
    Diff2 = [Cz[i]-2*Cz[i-1]+Cz[i-2] for i in range(2, len(Cz))]
    return Diff2

class GompertzI(OdeModel):
    """
    A class representing the type-I Gompertz ODE model.
    """
    @staticmethod
    def simulate_theta(t, theta):
        """
        Simulate the type-I Gompertz ODE model using the given time points and parameters.
        
        Parameters
        ----------
        t : array-like
            Time points at which to evaluate the model.
        theta : array-like
            Model parameters.

        Returns
        -------
        array-like
            Simulated values of the Gompertz I model at the given time points.
        """
        t = np.array(t)
        b, K, y0 = theta[:3]
        return K*(y0/K)**np.exp(-b*t)

class Grm(OdeModel):
    """
    A class representing the Generalized Richards Model (GRM) ODE model.
    """
    @staticmethod
    def simulate_theta(t, theta):
        """
        Simulate the GRM ODE model using the given time points and parameters.
        
        Parameters
        ----------
        t : array-like
            Time points at which to evaluate the model.
        theta : array-like
            Model parameters.

        Returns
        -------
        array-like
            Simulated values of the GRM model at the given time points.
        """
        t = np.array(t)
        r, alp, K, p, y0 = theta[:5]

        def der(y, t, r, alp, K, p):
            C = y
            dCdt = r*(C**p)*(1-(C/K)**alp)
            return dCdt
        yMod = odeint(der, y0, t, args=(r, alp, K, p))
        return yMod.ravel()

class EchtModifiedGompertz(OdeModel):
    """
    A class representing the EChT Modified Gompertz ODE model.
    """
    @staticmethod
    def simulate_theta(t, theta):
        """
        Simulate the EChT Modified Gompertz ODE model using the given time points and parameters.
        
        Parameters
        ----------
        t : array-like
            Time points at which to evaluate the model.
        theta : array-like
            Model parameters.

        Returns
        -------
        array-like
            Simulated values of the Echt Modified Gompertz model at the given time points.
        """
        t = np.array(t)
        alp, i, i0, bet, gam, y0 = theta[0:6]
    
        def mod_gomp(t, alp, i, i0, bet, gam, V0):
            a1 = (i/i0)*(2-i/i0)
            a2 = (1-i/i0)
            alpX = (a1*(1-np.exp(-gam*t))+a2)*alp
            VX = V0*np.exp((alpX/bet)*(1-np.exp(-bet*t)))
            return VX  
        
        yMod = mod_gomp(t, alp, i, i0, bet, gam, y0)
        yMod = np.array(yMod)
        return yMod.ravel()

class EchtVkt(OdeModel):
    """
    A class representing the EChT V-K-T ODE model.
    """
    @staticmethod
    def simulate_theta(t, theta, all_states=False):
        """
        Simulate the EChT V-K-T ODE model using the given time points and parameters.
        
        Parameters
        ----------
        t : array-like
            Time points at which to evaluate the model.
        theta : array-like
            Model parameters.
        all_states : bool, optional
            If True, return all state variables. If False, return only the first state variable.
            Default is False.

        Returns
        -------
        array-like
            Simulated values of the Echt Vkt model at the given time points.
        """
        t = np.array(t)
        b, c, d, u, v, V0, K0, T0 = theta[0:8]
        
        def der(y, t, bgomp, c, d, u, v):
            derN = bgomp*y[0]*np.log(y[1]/y[0])-y[2]*y[0]
            derK = c*y[0]-d*(y[0])**(2/3)*y[1]-u*y[0]*y[2]
            derT = -v*y[2]
            return [derN, derK, derT]  
        
        sol = odeint(der, (V0, K0, T0), t,
                     args=(b, c, d, u, v))
        sol = np.transpose(sol) 
        if all_states:
            return sol
        else:
            return sol[0].ravel()

class EchtVt(OdeModel):
    """
    A class representing the Echt Vt ODE model.
    """
    @staticmethod
    def simulate_theta(t, theta, all_states=False):
        """
        Simulate the Echt Vt ODE model using the given time points and parameters.
        
        Parameters
        ----------
        t : array-like
            Time points at which to evaluate the model.
        theta : array-like
            Model parameters.
        all_states : bool, optional, default=False
            If True, return all state variables in the output.

        Returns
        -------
        array-like
            Simulated values of the Echt Vt model at the given time points.
        """
        t = np.array(t)
        alpha, beta, v, V0, T0 = theta[0:5]

        def der(y, t, alpha, beta, v):
            derN = alpha*y[0]-beta*y[0]*np.log(y[0]/V0)-y[1]*y[0]
            derT = -v*y[1]
            return [derN, derT]

        sol = odeint(der, (V0, T0), t,
                     args=(alpha, beta, v))
        sol = np.transpose(sol)
        if all_states:
            return sol
        else:
            return sol[0].ravel()

        