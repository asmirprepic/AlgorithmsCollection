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

  def fit(self, X: List[List[float]],y: List[float] 
