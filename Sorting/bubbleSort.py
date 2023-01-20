

#### This file implements bubblesort

def bubbleSort(A):
  n=len(A)
  swapped = False
  
  # Loop from last to first
  for i in range(n-1,0,-1):
    for j in range(n):
      
      ## Swap if element is less than next element
      if A[j] > A[j+1]:
        swapped = True
        
        A[j],A[j+1]=A[j+1],A[j]
        
    if not swapped:
      ## Exit if not swapped
      
      return
    
return A
        
     
  
      
