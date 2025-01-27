from typing import List, Tuple, Dict, Union


import numpy as np

class NaiveBayes:

  """ Naive bayes classification implementation. 
      Supports GNB for continuous features. 
  """

  def __init__(self):
    """  
      Initialize the naive bayes
    """

    self.priors: Dict[str,float] = {}
    self.means: Dict[str,np.ndarray] = {}
    self.stds: Dict[str, np.ndarray] = {}

  def fit(self, X: List[List[float]],y: List[float] -> List[str]:
     """
        Trains the Naive Bayes classifier on the given data.

        Args:
            X: Training data, where each element is a list of features.
            y: Labels for the training data.
      
    """
    classes, class_counts = np.unique(y, return_counts = True)
    total_samples = len(y)

    "Priors"

    self.priors ={cls: count/total_samples for cls, count in zip(classes,class_counts)}
    self.means = {}
    self.stds ={}

    for cls in classes: 
      X_cls = np.array([X[i] for i in range(len(y)) if y[i] == cls)
      self.means[cls] = np.mean(X_cls,axis = 0)
      self.stds[cls] = np.std[X_cls,axis = 0)


  def predict(self, X_test: List[List[float]]) -> List[str]:
    """  
      Predicts the class labels for given test data

      Args: 
        X_test: test data, each element is a list of features

      Returns: 
        A list of predicted class labels
    """
    predictions = []
    for x in X_test: 
      class_probs = {}
      for cls in self.priors: 
        prior = self.priors[cls]
        likelihood = np.prod(self._guassian_pdf(x, self.means[cls],self.stds[cls]))
        class_probs[cls] = prior*likelihood
      predictions.append(max(class_probs, key = class_probs.get))

 def _gaussian_pdf(self, x: List[float], mean: np.ndarray, std: np.ndarray) -> List[float]:
        """
        Calculates the probability density function of a Gaussian distribution.

        Args:
            x: Input data point.
            mean: Mean of the Gaussian distribution.
            std: Standard deviation of the Gaussian distribution.

        Returns:
            A list of probability densities for each feature.
        """
        probs = []
        for xi, mu, sigma in zip(x, mean, std):
            if sigma > 0:  
                probs.append((1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xi - mu) / sigma)**2))
            else:
                
                probs.append(1.0 if xi == mu else 0.0)
        return probs
                             
                             
    
    



