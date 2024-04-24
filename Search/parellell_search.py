"""
An implementation of parallell search in python
"""

import multiprocessing

def search_chunk(chunk, query, output, index):
    """Searches for the query in the chunk and returns the result in a shared dictionary."""
    output[index] = query in chunk


def parallel_search(lst, query):
    n_processes = multiprocessing.cpu_count()  # Get the number of cores
    chunk_size = len(lst) // n_processes       # Determine the size of each chunk

    # Create a manager dictionary to store the results
    manager = multiprocessing.Manager()
    output = manager.dict()
    
    # List to hold all the processes
    processes = []

    # Create and start a process for each chunk
    for i in range(n_processes):
        start = i * chunk_size
        # Ensure the last process gets the remainder of the list
        end = None if i == n_processes - 1 else start + chunk_size
        chunk = lst[start:end]
        p = multiprocessing.Process(target=search_chunk, args=(chunk, query, output, i))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Check if any process found the element
    return any(output.values())

# Example data
data_list = list(range(1000000))  # A large list of numbers
search_query = 999999             # Element we're searching for

# Perform the search
found = parallel_search(data_list, search_query)
print("Element found:", found)

