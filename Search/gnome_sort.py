import numpy as np

def gnome_sort(arr):
    idx = 0
    while idx < len(arr):
        if idx == 0 or arr[idx] >= arr[idx - 1]:
            idx += 1
        else:
            arr[idx], arr[idx - 1] = arr[idx - 1], arr[idx]
            idx -= 1
    return arr

# Example usage
arr = np.array([34, 2, 10, -9, 7, 1, 3])
sorted_arr = gnome_sort(arr)
print("Sorted array:", sorted_arr)
