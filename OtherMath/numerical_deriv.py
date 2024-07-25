import numpy as np

class NumericalDerivative: 
    def __init__(self,function):
        """
        Initialize the numerical derivative class with a function.
        
        Parameters: 
        function (callable): the function to handle derivatives for
        """

        self.function = function

    def forward_difference(self,x,idx = None,h = 1e-5):
        """
        Calculate the the forward difference derivative of a function at point x. 

        Parameters: 
        x (float or np.array): the forward difference derivative at point x. 
        idx (int): the index of the variable with respect to which to take the derivative (for multivariate functions)
        h (float): The step size for the finite diffrence calculation.

        Returns: 
        flot or numpy.ndarray: The forward diffrence derivative at point x.
        """

        if np.isscalar(x):
            return (self.function(x+h)-self.function(x))/h
        
        else:
            x = np.asarray(x)
            if idx is None: 
                return (self.function(x+h)-self.function(x))/h
            else:
                x_h = x.copy()
                x_h[:,idx] += h
                return (self.function(x_h) + self.function(x))/h
    
    def backward_difference(self,x,idx = None, h = 1e-5):
        """
        Calculate the the backward difference derivative of a function at point x. 

        Parameters: 
        x (float or np.array): the forward difference derivative at point x. 
        idx (int): the index of the variable with respect to which to take the derivative (for multivariate functions)
        h (float): The step size for the finite diffrence calculation.

        Returns: 
        flot or numpy.ndarray: The backward diffrence derivative at point x.
        """

        if np.isscalar(x):
            return (self.function(x)-self.function(x-h))/h
        
        else:
            x = np.asarray(x)
            if idx is None: 
                return (self.function(x)-self.function(x-h))/h
            else:
                x_h = x.copy()
                x_h[:,idx] -= h
                return (self.function(x) + self.function(x_h))/h
    
    def central_difference(self,x,idx = None, h = 1e-5):
        """
        Calculate the the central difference derivative of a function at point x. 

        Parameters: 
        x (float or np.array): the forward difference derivative at point x. 
        idx (int): the index of the variable with respect to which to take the derivative (for multivariate functions)
        h (float): The step size for the finite diffrence calculation.

        Returns: 
        flot or numpy.ndarray: The central diffrence derivative at point x.
        """

        if np.isscalar(x):
            return (self.function(x+h)-self.function(x-h))/(2*h)
        
        else:
            x = np.asarray(x)
            if idx is None: 
                return (self.function(x+h)-self.function(x-h))/(2*h)
            else:
                x_h1= x.copy()
                x_h2 = x.copy()
                x_h1[:,idx] += h
                x_h2[:,idx] -=h
                return (self.function(x_h1) + self.function(x_h2))/h
    
    def gradient(self,x,h=1e-5):
        """
        Calculate the gradient of the function at point x (for multivariale functions)

        Parameters: 
            x (np.ndarray): The point at which to evaluate the gradient
            h (float): The step size for the finite difference calculation

            Returns np.ndarray: The gradient at point x.
        """

        grad = np.zeros_like(x)
        for idx in range(x.shape[1]):
            grad[:,idx ] = self.central_difference(x,idx,h)
        return grad