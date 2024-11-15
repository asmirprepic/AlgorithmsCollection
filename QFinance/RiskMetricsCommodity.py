class RiskMetrics:
    def __init__(self, shocked_prices):
        """
        Initialize with shocked prices to analyze P&L and risk metrics.
        """
        self.shocked_prices = shocked_prices

    def calculate_pnl(self, base_prices):
        """
        Calculate P&L based on difference between shocked prices and base prices.
        """
        pnl = {commodity: shocked_prices - base_prices[commodity] for commodity, shocked_prices in self.shocked_prices.items()}
        return pnl

    def calculate_var(self, pnl_data, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
        - pnl_data: P&L data after applying shocks.
        - confidence_level: Confidence level for VaR calculation.
        """
        var_results = {}
        cvar_results = {}

        for commodity, pnl in pnl_data.items():
            sorted_pnl = np.sort(pnl)
            var_index = int((1 - confidence_level) * len(sorted_pnl))
            var_results[commodity] = sorted_pnl[var_index]
            cvar_results[commodity] = sorted_pnl[:var_index].mean()
            logging.info("Calculated VaR and CVaR for %s at %s confidence level.", commodity, confidence_level)
        
        return var_results, cvar_results
