

class KMeans:
  def __init__(self,k = 3, max_iterations = 100):
    self.k = k
    self.max_iterations = max_iterations

  def fit(self,X):
    self.centroids = X[np.random.choice(X.shape[0],self.k,replace = False)

    for _ in range(self.centroids):
      labels = np.zeros(X.shape[0])
      
