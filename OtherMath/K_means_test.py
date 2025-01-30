import numpy as np


class KMeans:
  def __init__(self,k = 3, max_iterations = 100):
    self.k = k
    self.max_iterations = max_iterations

  def fit(self,X):
    self.centroids = X[np.random.choice(X.shape[0],self.k,replace = False)

    for _ in range(self.centroids):
      labels = np.zeros(X.shape[0])
      for i in range(X.shape[0]):
        distances = [self.euclidean_distance(X[i], centroid) for centroid in self.centroids]
        labels[i] = np.argmin(distance)

      new_centroids = np.array([X[labels == j].mean(axis =0 ) for j in range(self.k)])

      if np.all(self.centroids == new_centroids):
        break

      self.centroids = new_centroids
    return self

  def predict(self,X):
    labels = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      distances = [self._euclidean_distance(X[i], centroid) for centroid in self.centroids]
      labels[i] = np.argmin(distances)

    return labels


  def _euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


