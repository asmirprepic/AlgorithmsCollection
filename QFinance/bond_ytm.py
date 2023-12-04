"""
Illustrationg of cacluating the bonds yield to maturity
"""

import scipy.optimize as optimize

def bond_ytm(price,par,T,coup,freq=2,guess=0.5):
    freq=float(freq)
    periods = T*2
    coupon= coup/100.*par
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y: sum([coupon/freq/(1+y/freq)**(freq*t) for t in dt]) +  par/(1+y/freq)**(freq*T) - price
    
    return optimize.newton(ytm_func,guess)
