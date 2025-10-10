import math

def logpdf_norm(x, mu, sigma):
    s2 = sigma*sigma
    return -0.5*math.log(2*math.pi*s2) - 0.5*(x-mu)**2/s2

def logpdf_student_t(x, df, loc, scale):
    # Student-t log pdf via gamma functions
    v = df; z = (x - loc) / scale
    return (math.lgamma((v+1)/2) - math.lgamma(v/2)
            - 0.5*math.log(v*math.pi) - math.log(scale)
            - ((v+1)/2)*math.log(1 + z*z/v))

class SequentialLLR:
    """
    One-pass sequential log-likelihood ratio:
      llr_t += log f1(x_t) - log f0(x_t)
    Exposes:
      - BF (Bayes factor), BF_sup (running max), p_anytime = 1/BF_sup
      - Optional Wald SPRT thresholds A,B for quick decisions.
    """
    def __init__(self, logpdf0, logpdf1, sprt_alpha=None, sprt_beta=None):
        self.logpdf0 = logpdf0
        self.logpdf1 = logpdf1
        self.llr = 0.0
        self.n = 0
        self.BF = 1.0
        self.BF_sup = 1.0
        self.A = self.B = None
        if sprt_alpha is not None and sprt_beta is not None:
            # Wald boundaries on LLR (log-space)
            self.A = math.log((1 - sprt_beta) / sprt_alpha)   # accept H1 if LLR >= A
            self.B = math.log(sprt_beta / (1 - sprt_alpha))   # accept H0 if LLR <= B
        self.decision = None  # "H1", "H0", or None

    def update(self, x):
        if self.decision is not None:
            return self.state()
        self.n += 1
        self.llr += self.logpdf1(x) - self.logpdf0(x)
        self.BF = math.exp(self.llr)
        self.BF_sup = max(self.BF_sup, self.BF)
        if self.A is not None:
            if self.llr >= self.A: self.decision = "H1"
            elif self.llr <= self.B: self.decision = "H0"
        return self.state()

    def state(self):
        return {
            "n": self.n,
            "LLR": self.llr,
            "BF": self.BF,
            "BF_sup": self.BF_sup,
            "p_anytime": 1.0 / self.BF_sup,
            "decision": self.decision
        }
