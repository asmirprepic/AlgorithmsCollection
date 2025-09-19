import math

class CoinBetMartingale:
    """
    KT coin-betting martingale for ±1 data. Wealth W_t is a nonnegative martingale under H0 (fair coin).
    Two-sided version runs betting on X and on -X, takes the max wealth for a two-sided p-value.
    """
    def __init__(self):
        self.n = 0
        self.S = 0.0          # cumulative sum of x_t
        self.W_pos = 1.0      # wealth betting that mean > 0
        self.W_neg = 1.0      # wealth betting that mean < 0
        self.W_sup = 1.0      # running sup of max wealth (for anytime p)

    def update(self, x: int):
        """
        Feed x in {-1, +1}. Returns (W, p_anytime), where W = max(W_pos, W_neg).
        """
        if x not in (-1, 1):
            raise ValueError("x must be +1 or -1")
        self.n += 1

        # KT fraction: f_t = S_{t-1} / t
        f = self.S / self.n
        f = max(min(f, 0.99), -0.99)

        self.W_pos *= (1.0 + f * x)
        self.W_neg *= (1.0 - f * x)

        # Update cumulative
        self.S += x

        # Two-sided wealth and anytime-valid p-value
        W = max(self.W_pos, self.W_neg)
        self.W_sup = max(self.W_sup, W)
        p_anytime = 1.0 / self.W_sup
        return W, p_anytime

    def reset(self):
        self.__init__()
