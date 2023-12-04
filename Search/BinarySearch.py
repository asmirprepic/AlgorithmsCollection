"""
This function implements a binary search 
"""
def binarySearch(A,e):
  first = 0
  last = len(A)-1
  index = -1
   
  while(first <= last) and (index==-1):
    mid = (first + last)//2
    if A[mid] == e:
      index = mid
    else:
      if e<A[mid]:
        last = mid -1
      
      else: 
        first = mid +1
  return index

print(binarySearch([2,2,2,2,2,3,2,2],3))  




