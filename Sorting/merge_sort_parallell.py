import concurrent.futures

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def parallel_merge_sort(arr, max_workers=None):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_left = executor.submit(parallel_merge_sort, left, max_workers)
        future_right = executor.submit(parallel_merge_sort, right, max_workers)
        
        left = future_left.result()
        right = future_right.result()

    return merge(left, right)

# Example usage
if __name__ == "__main__":
    import random

    array_size = 100
    arr = [random.randint(0, 1000) for _ in range(array_size)]
    
    print("Original array:", arr)
    sorted_arr = parallel_merge_sort(arr)
    print("Sorted array:", sorted_arr)
