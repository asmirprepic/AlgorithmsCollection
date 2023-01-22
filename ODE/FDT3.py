## Finite difference example for y'' = -4y + 4x , y(0) = 0, y'(pi/2) = 0

def FDequations(x,h,m):
  h2 = h*h
  d = np.ones(m+1)*(-2+4.0*h2)
  c = np.ones(m)
  e = np.ones(m)
  b = np.ones(m+1)*4.0*h2*x
  d[0] = 1.0
  e[0] = 0.0
  b[0] = 0.0
  c[m-1] = 2.0
  return c,d,e,b


## Using LU decomposition to solve

def LUdecomp(c,d,e):
  n = len(d)

  for k in range(1,n):
    lam = c[k-1]/d[k-1]
    d[k] = d[k] - lam*e[k-1]
    c[k-1] = lam
  
  return c,d,e

## Solving Tridiagonal LU decomposed matrix
def LUSolve(c,d,e,b):
  n = len(d)
  for k in range(1,n):
    b[k] =b[k]-c[k-1]*b[k-1]
  
  b[n-1] = b[n-1]/d[n-1]

  for k in range(n-2,-1,-1):
    b[k] = (b[k]- e[k]*b[k+1])/d[k]
  
  return b


## Solving 

xstart = 0.0
xstop  = math.pi/2.0
m=10
h = (xstop-xstart)/m
x = np.arange(xstart,xstop+h,h)
c,d,e,b = FDequations(x,h,m)
c,d,e = LUdecomp(c,d,e)
y = LUSolve(c,d,e,b)



plt.plot(x,y)
plt.grid(True)
plt.show()
  

