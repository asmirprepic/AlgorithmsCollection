def custom_kalman_filter(self, observations, q, r):
        """
        Custom 1D Kalman filter implementation.

        Args:
            observations (pd.Series): Price observations.
            q (float): Process variance.
            r (float): Measurement variance.

        Returns:
            np.array: Estimated trend.
            np.array: Estimation standard deviation.
        """
        n = len(observations)
        x_est = np.zeros(n)
        P = np.zeros(n)
        K = np.zeros(n)

        # Initialize estimates
        x_est[0] = observations.iloc[0]
        P[0] = 1.0

        for t in range(1, n):
            # Prediction
            x_pred = x_est[t-1]
            P_pred = P[t-1] + q

            # Kalman Gain
            K[t] = P_pred / (P_pred + r)

            # Update
            x_est[t] = x_pred + K[t] * (observations.iloc[t] - x_pred)
            P[t] = (1 - K[t]) * P_pred

        return x_est, np.sqrt(P)
