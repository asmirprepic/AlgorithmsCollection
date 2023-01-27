### Fibonacci search ###
def fibonacciSearch(A,e):
  firstElement = 0
  secondElement = 1

  fibonacciElement = firstElement + secondElement

  while(fibonacciElement < len(A)):
    firstElement = secondElement
    secondElement = fibonacciElement
    fibonacciElement = firstElement + secondElement
  index =-1

  while (fibonacciElement > 1):
    i = min(index + firstElement,(len(A)-1))
    if(A[i]<e):
      fibonacciElement = secondElement
      secondElement = firstElement
      firstElement = fibonacciElement- secondElement
      index = i
    elif(A[i] > e):
      fibonacciElement = firstElement
      secondElement = secondElement - firstElement
      firstElement = fibonacciElement - secondElement

    else: 
      return i

  if(secondElement and index < (len(A)-1) and A[index + 1]==e):
    return index + 1
  return -1

print(fibonacciSearch([1,2,3,4,5,6,7,8,9,10,11], 7))  

