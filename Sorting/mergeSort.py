def mergeSort(A):
  if len(A)>1:

    #Find the middle
    mid = len(A)//2

    L = A[:mid]
    R = A[mid:]

    #Sort first half  
    mergeSort(L)

    #Sort second half
    mergeSort(R)

    i=j=k=0

    while i < len(L) and j < len(R):
      if L[i] < R[i]:
        A[k] = L[i]
        i += 1
      else:
        A[k] = R[j]
        j += 1

      k += 1

    while i < len(L):
      A[k] = L[i]
      i += 1
      k += 1

    while j < len(R):
      A[k] = R[j]
      j += 1
      k += 1
    return A

        
        
