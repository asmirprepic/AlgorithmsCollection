def _e_step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expectation step: forward-backward algorithm."""
        alpha = np.zeros((self.T, 2))
        beta = np.zeros((self.T, 2))
        scale = np.zeros(self.T)

        # Forward recursion
        for k in range(2):
            alpha[0, k] = self.pi[k] * self._gaussian_density(self.returns[0], self.mu[k], self.sigma2[k])
        scale[0] = np.sum(alpha[0])
        alpha[0] /= scale[0]

        for t in range(1, self.T):
            for k in range(2):
                alpha[t, k] = self._gaussian_density(self.returns[t], self.mu[k], self.sigma2[k]) * np.sum(
                    alpha[t - 1] * self.P[:, k]
                )
            scale[t] = np.sum(alpha[t])
            alpha[t] /= scale[t]

        # Backward recursion
        beta[-1] = 1
        for t in reversed(range(self.T - 1)):
            for k in range(2):
                beta[t, k] = np.sum(
                    beta[t + 1] * self._gaussian_density(self.returns[t + 1], self.mu, self.sigma2) * self.P[k]
                )
            beta[t] /= np.sum(beta[t])  # normalize

        # Compute gamma and xi
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((self.T - 1, 2, 2))
        for t in range(self.T - 1):
            denom = 0
            for i in range(2):
                for j in range(2):
                    xi[t, i, j] = (
                        alpha[t, i]
                        * self.P[i, j]
                        * self._gaussian_density(self.returns[t + 1], self.mu[j], self.sigma2[j])
                        * beta[t + 1, j]
                    )
                    denom += xi[t, i, j]
            xi[t] /= denom

        return gamma, xi, scale

  def _m_step(self, gamma: np.ndarray, xi: np.ndarray) -> None:
      """Maximization step: update parameters."""
      for k in range(2):
          weights = gamma[:, k]
          self.mu[k] = np.sum(weights * self.returns) / np.sum(weights)
          self.sigma2[k] = np.sum(weights * (self.returns - self.mu[k]) ** 2) / np.sum(weights)

      self.P = np.sum(xi, axis=0)
      self.P /= self.P.sum(axis=1, keepdims=True)
      self.pi = gamma[0]

  def fit(self) -> None:
      """Run the EM algorithm to estimate parameters."""
      log_likelihood_old = -np.inf

      for i in range(self.n_iter):
          gamma, xi, scale = self._e_step()
          self._m_step(gamma, xi)

          log_likelihood = np.sum(np.log(scale))
          if abs(log_likelihood - log_likelihood_old) < self.tol:
              break
          log_likelihood_old = log_likelihood

      self.gamma = gamma  # Store responsibilities
      self.log_likelihood = log_likelihood
