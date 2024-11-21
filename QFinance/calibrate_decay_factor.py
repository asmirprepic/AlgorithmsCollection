def calibrate_decay_factor(returns, realized_variance):
    """
    Calibrate decay factor to minimize MAPE between forecasted and realized variance.
    """
    def objective(decay_factor):
        forecast = ewma_forecast(returns, decay_factor)
        return mean_absolute_percentage_error(realized_variance[~np.isnan(realized_variance)], forecast[~np.isnan(realized_variance)])
    
    from scipy.optimize import minimize
    result = minimize(objective, x0=0.94, bounds=[(0.80, 0.99)])
    return result.x[0]
