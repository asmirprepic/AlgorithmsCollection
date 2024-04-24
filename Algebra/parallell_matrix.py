"""
Parallell matrix multiplication in python
"""
import multiprocessing 

def matrix_multiply(row, B):
    return [sum(x * y for x, y in zip(row, col)) for col in zip(*B)]

import multiprocessing

def parallel_matrix_multiply(A, B):
    # Function to be executed in parallel
    def worker(row, output, index):
        # Compute the multiplication of the row by matrix B
        result_row = matrix_multiply(row, B)
        # Store the result in the output dictionary with the index as key
        output[index] = result_row

    # Creating a dictionary to store the results
    manager = multiprocessing.Manager()
    output = manager.dict()

    # List to keep track of processes
    processes = []

    # Create a process for each row in matrix A
    for i, row in enumerate(A):
        # Args: row of matrix A, shared dict, index of the row
        p = multiprocessing.Process(target=worker, args=(row, output, i))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Collect the results from the output dictionary and sort by keys to maintain row order
    result_matrix = [output[i] for i in sorted(output)]

    return result_matrix

A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]

# Multiply the matrices in parallel
result = parallel_matrix_multiply(A, B)
print(result)
