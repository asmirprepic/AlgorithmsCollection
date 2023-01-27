import numpy as np

## Linear Search ## 

def linearSearch(A,e):
  for i in range(len(A)):
    if A[i] == e:
      return i
  
  return -1

### testing ###
print(linearSearch([2,2,2,2,2,3,2,2],3))  


### enumerate method

def lineraSerach(A,e):
 pos = [i if i == e else -1 for i in A]
 
 return pos

print(linearSearch([2,2,2,2,2,3,2,2],3))  



