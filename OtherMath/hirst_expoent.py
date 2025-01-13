def hurst_exponent(time_series, min_window=10, max_window=None, num_scales=10):
    """
    Computes the Hurst exponent using Rescaled Range (R/S) analysis.
    
    Parameters:
        time_series (np.ndarray): 1D array of time series data.
        min_window (int): Minimum segment length for R/S analysis.
        max_window (int): Maximum segment length for R/S analysis (default: 1/4 of series length).
        num_scales (int): Number of log-spaced scales for analysis.

    Returns:
        float: Estimated Hurst exponent.
    """
    if max_window is None:
        max_window = len(time_series) // 4
    
    window_sizes = np.logspace(np.log10(min_window), np.log10(max_window), num_scales, dtype=int)
    rescaled_ranges = []
    
    for window in window_sizes:
        num_segments = len(time_series) // window
        R_S_values = []
        
        for i in range(num_segments):
            segment = time_series[i * window : (i + 1) * window]
            if len(segment) < window:
                continue
            
            Y = segment - np.mean(segment)  
            X = np.cumsum(Y)  
            
            R = np.max(X) - np.min(X)  
            S = np.std(segment)  
            
            if S > 0:  
                R_S_values.append(R / S)
        
        if len(R_S_values) > 0:
            rescaled_ranges.append(np.mean(R_S_values))

    
    log_sizes = np.log(window_sizes[:len(rescaled_ranges)])
    log_RS = np.log(rescaled_ranges)

    slope, _, _, _, _ = linregress(log_sizes, log_RS)
    
    return slope  
