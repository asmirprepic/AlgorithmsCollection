import math

class VarCUSUM:
    """
    Detect σ^2 shift: H0: σ^2=σ0^2  vs  H1: σ^2=c*σ0^2.
    Two-sided: run 'up' (c>1) and 'down' (c<1) stats in parallel.
    """
    def __init__(self, sigma0: float, c_up=2.0, c_dn=0.5, h=10.0):
        self.s02 = float(sigma0)**2
        self.cu, self.cd = float(c_up), float(c_dn)
        self.h = float(h)
        self.Cu = 0.0; self.Cd = 0.0

    def update(self, x: float):
        z2 = (float(x)**2) / self.s02
        # LLR increments (Gaussian, mean 0)
        lu = 0.5 * (z2*(1/self.cu - 1) + math.log(self.cu))
        ld = 0.5 * (z2*(1/self.cd - 1) + math.log(self.cd))
        self.Cu = max(0.0, self.Cu + lu)
        self.Cd = max(0.0, self.Cd + ld)
        up = self.Cu > self.h
        dn = self.Cd > self.h
        if up or dn: self.Cu = self.Cd = 0.0
        return {"Cu": self.Cu, "Cd": self.Cd, "alarm_up": up, "alarm_dn": dn}
