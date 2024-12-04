def price_dividend_swap(dividends, notional, r, T):
    """
    Price a dividend swap using stochastic dividends.
    """
    avg_dividends = dividends[:int(T * 12)].mean(axis=1)
    pv_dividends = np.sum(avg_dividends * np.exp(-r * np.arange(1, T * 12 + 1) / 12))
    return notional * pv_dividends
