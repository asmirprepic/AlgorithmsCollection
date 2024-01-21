def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def count_derangements(n):
    if n == 0:
        return 1
    elif n == 1:
        return 0
    else:
        return (n - 1) * (count_derangements(n - 1) + count_derangements(n - 2))

def probability_no_match_exact(n):
    return count_derangements(n) / factorial(n)

# Example usage
num_people = 10
probability_exact = probability_no_match_exact(num_people)
print(f"Probability that no one gets their own hat (exact): {probability_exact}")
