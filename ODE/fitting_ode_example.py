from lmfit import minimize, Parameters, Paremeter,repot
from scipy.integrate import odeint

def (xs,t,ps):
"""Simple model"""
  try: 
    a = ps['a'].value
    b = ps['b'].value
  except: 
    a,b = ps
  x = xs
  return a -bx

def sol_ode(t,x0,ps):
  x = odeint(f,x0,t,args=(ps,))
  return x

def residual(ps,ts,data):
  x0 = ps['x0'].value
  model = sol_ode(ts,x0,ps)
  return (model-data)
  
