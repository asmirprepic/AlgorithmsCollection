def expected_flips(n):
    if n == 0:
        return 0
    else:
        return 2 + expected_flips(n - 1)

# Calculate the expected number of flips to get 3 heads
expected_number_of_flips = expected_flips(3)
print(f"The expected number of flips until the third heads appears is {expected_number_of_flips}")
