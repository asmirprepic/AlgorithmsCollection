import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class CopulaFitter:
    def __init__(self, data, copula_type='gaussian', df=None):
        """
        Initialize the copula fitter with bivariate data and copula type.
        """
        self.data = data
        self.copula_type = copula_type.lower()
        self.df = df if copula_type == 't' else None
        self.u_data = None
        self.v_data = None
        self.correlation_matrix = None
    
    def empirical_cdf(self, data):
        
        ranks = np.argsort(np.argsort(data))
        uniform_data = (ranks + 1) / (len(data) + 1)  
        return uniform_data
    
    def transform_to_uniform(self):
        """
        Transform both marginals of the bivariate data to uniform distribution.
        """
        self.u_data = self.empirical_cdf(self.data[:, 0])
        self.v_data = self.empirical_cdf(self.data[:, 1])
    
    def fit(self):
        """
        Fit a copula to the transformed data based on the chosen copula type.
        """
        if self.u_data is None or self.v_data is None:
            self.transform_to_uniform()
        
        
        uniform_data = np.vstack([self.u_data, self.v_data]).T
        
        if self.copula_type == 'gaussian':
        
            z_data = stats.norm.ppf(uniform_data)
        elif self.copula_type == 't':
            if self.df is None:
                raise ValueError("Degrees of freedom (df) must be provided for t-Copula.")
        
            z_data = stats.t.ppf(uniform_data, df=self.df)
        else:
            raise ValueError(f"Unsupported copula type: {self.copula_type}")
        
        
        self.correlation_matrix = np.corrcoef(z_data.T)
        
        return self.correlation_matrix
    
    def plot(self):
        """
        Plot the original data and the transformed uniform data.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        
        axs[0].scatter(self.data[:, 0], self.data[:, 1], alpha=0.6)
        axs[0].set_title("Original Bivariate Data")
        axs[0].set_xlabel("X1")
        axs[0].set_ylabel("X2")
        
        
        axs[1].scatter(self.u_data, self.v_data, alpha=0.6)
        axs[1].set_title(f"Transformed Uniform Data ({self.copula_type.capitalize()} Copula)")
        axs[1].set_xlabel("U1")
        axs[1].set_ylabel("U2")
        
        plt.tight_layout()
        plt.show()
    
    def display_correlation_matrix(self):
        """
        Print the fitted copula correlation matrix.
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix has not been computed. Call `fit()` first.")
        
        print(f"Fitted {self.copula_type.capitalize()} Copula Correlation Matrix:")
        print(self.correlation_matrix)


# Example Usage
if __name__ == "__main__":
    
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]  
    data = np.random.multivariate_normal(mean, cov, size=500)

    
    gaussian_copula = CopulaFitter(data, copula_type='gaussian')
    gaussian_copula.fit()
    gaussian_copula.display_correlation_matrix()
    gaussian_copula.plot()

    
    t_copula = CopulaFitter(data, copula_type='t', df=5)
    t_copula.fit()
    t_copula.display_correlation_matrix()
    t_copula.plot()
