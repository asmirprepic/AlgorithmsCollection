

def insertionsort(A):
  
  ## Loop from second to last
  for i in range(1,len(A)):
    
    ##Item to be placed
    key_item = A[i]
    
    # variable to find position
    j=i-1
    
    ## Running the list in items
    while j>=0 and A[j] > key_item:
      
      A[j+1] = A[j]
      j-=1
      
    ## When you ins
    
    A[j+1] = key_item
    
return A
    
    
  
