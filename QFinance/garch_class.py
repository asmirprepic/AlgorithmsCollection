import numpy as np
from scipy.optimize import minimize


class GARCH:
    def __init__(self,returns,p=1,q=1):
        self.returns = np.asarray(returns)
        self.p = p
        self.q = q
        self.n = len(self.returns)
        self.parameters = None
        self.omega = None
        self.alpha = None
        self.beta = None
        self.variances = None

    def log_likelihood(self,params):
        omega = params[0]
        alpha = params[1:self.p+1]
        beta = params[self.p+1:self.p+self.q+1]

        # Initialize the variance series
        var = np.zeros_like(self.returns)
        var[:max(self.p,self.q)] = np.var(self.returns) # Set initial variance

        # Calculate the conditional variances using the GARCH(p, q) model
        for t in range(max(self.p,self.q),self.n):
            var[t] = omega
            for i in range(1,self.p+1):
                # Add ARCH terms: alpha_i * r_{t-i}^2
                var[t] += alpha[i-1]*self.returns[t-i]**2
            for j in range(self.q+1):
                # Add GARCH terms: beta_j * sigma_{t-j}^2
                var[t] += beta[j-1] * var[t-j]
        
        # Log-likelihood calculation
        # The log-likelihood for the GARCH model is:
        # L = -0.5 * sum(log(sigma_t^2) + (r_t^2 / sigma_t^2))
        log_likelihood = -0.5*np.sum(np.log(var) + self.returns**2/var) 
        return -log_likelihood
    
    def fit(self):
        initial_params = [0.1] + [0.1]*self.p + [0.8]*self.q
        # Parameter bounds: omega > 0, 0 < alpha_i < 1, 0 < beta_j < 1, and sum(alpha_i + beta_j) < 1   
        bounds = [(1e-6,None)] + [(1e-6,1-1e-6)]*(self.p+self.q)

        result = minimize(self.log_likelihood,initial_params,bounds=bounds,method = 'L-BFGS-B')

        if result.success:
            self.parameters = result.x
            self.omega = self.parameters[0]
            self.alpha = self.parameters[1:self.p+1]
            self.beta = self.parameters[self.p+1:self.p+self.q+1]
            self.calculate_variances()
        
        else:
            raise ValueError("Optimization failed. Try different initial parameters or check the data")
        
    def calculate_variances(self):
        self.variances = np.zeros_like(self.returns)
        self.variances[:max(self.p,self.q)] = np.var(self.returns)

        for t in range(max(self.p,self.q),self.n):
            self.variances[t] = self.omega

            for i in range(1,self.p+1):
                self.variances[t] += self.alpha[i-1]*self.returns[t-i]**2
            for j in range(1,self.q+1):
                self.variances[t] += self.beta[j-1]*self.variances[t-j]
    
    def predict(self,steps = 10):
        forecast_var = np.zeros(steps)
        forecast_var[0] = self.omega
        for i in range(1,self.p+1):
            forecast_var[0] += self.alpha[i-1]*self.returns[-i]**2
        for j in range(1,self.q+1):
            forecast_var[0] += self.beta[j-1]*self.variances[-j]
        
        for t in range(1,steps):
            forecast_var[t] = self.omega
            
            for i in range(1,self.p+1):
                if t-i >= 0:
                    forecast_var[t] += self.alpha[i-1]*forecast_var[t-i]
            for j in range(1,self.q+1):
                if t-j >= 0:
                    forecast_var[t] += self.beta[j-1]*forecast_var[t-j]
        return forecast_var
    
    def get_volatility(self):
        if self.variances is None: 
            raise ValueError("Model must be fitted before calculating volatility")
        return np.sqrt(self.varainces)
    
    def summary(self):
        if self.parameters is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'log_likelihood': -self.log_likelihood(self.parameters)
        }
