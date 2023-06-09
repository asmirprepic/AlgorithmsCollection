def zero_coupon_bond(par,y,t):
  """
  Price of a zero coupon bond
  par: face value of the bond
  y: annual yield
  t: time to maturity
  """
  
  return par/(1+y)**t
