import matplotlib.pylot as plt
import numpy as np

class DDE:
  def __init__(self,deriv,delays,history,t0 = 0):
    self.deriv = deriv
    self.delays = delays
    self.history = history
    self.t0 = t0

  def solve(self,t_end,dt):
    self.steps = int(np.ceil((t_end -self.t0)/dt))
    self.times = np.linspace(self.t0,t_end,self.steps +1)
    self.solution = np.zeros(self.steps +1)
    self.past_values = [(self.t0-d,self.history(self.t0-d)) for d in np.linspace(0,max(self.delays),100)]

  def get_delayed_value(t):
    if t <= self.t0:
      return self.history(t)
    else: 
      interp_times, interp_values = zip(*self.past_values)
      interpolator = interpr1d(interp_times,interp_values, kind = 'cubic', fill_values = 'extrapolate')
      return interpolator(t)
