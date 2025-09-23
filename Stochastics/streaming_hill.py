import heapq, numpy as np

class StreamingHill:
    """
    Keep top-k absolute losses; Hill estimator γ = (1/k) Σ log(X_(i)/X_(k+1)).
    Use for tail heaviness monitoring (larger γ ~ heavier tail).
    """
    def __init__(self, k=100):
        self.k = int(k)
        self.heap = []  # min-heap of top-k magnitudes

    def update(self, x: float) -> float:
        v = abs(float(x))
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, v)
        elif v > self.heap[0]:
            heapq.heapreplace(self.heap, v)
        if len(self.heap) < self.k:
            return np.nan
        xs = np.sort(np.array(self.heap))
        xk = xs[0]  # (k-th largest as smallest in heap)

        if xk <= 0: return np.nan
        logs = np.log(xs / xk + 1e-12)
        gamma = logs.mean()
        return float(gamma)
