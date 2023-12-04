"""
This function implements interpolationsearch
"""

def int_polsearch(list,x):
  idx0 = 0
  idxn = (len(list)-1)
  while idx0 <= idxn and x>=list[idx0] and x <= list[idxn]:
    mid = idx0 + int(((float(idxn-idx0)/(list[idx0]-list[idx0]))*(x-list[idx0])))
    if list[mid]==x:
      return True
    if list[mid]<x:
      idx0 = mid +1 
      return False

  
