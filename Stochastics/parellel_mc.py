import multiprocessing as mp
import numpy as np

def monte_carlo_pi_part(samples):
    np.random.seed()  # Ensure each process has a different seed
    count = 0
    for _ in range(samples):
        x, y = np.random.uniform(0, 1, 2)
        if x**2 + y**2 <= 1:
            count += 1
    return count

def parallel_monte_carlo_pi(total_samples, num_processes):
    pool = mp.Pool(processes=num_processes)
    samples_per_process = total_samples // num_processes

    # Divide the work among processes
    counts = pool.map(monte_carlo_pi_part, [samples_per_process] * num_processes)

    pool.close()
    pool.join()

    # Aggregate results from all processes
    total_count = sum(counts)
    pi_estimate = (4 * total_count) / total_samples
    return pi_estimate

if __name__ == "__main__":
    total_samples = 10**7
    num_processes = mp.cpu_count()  # Use the number of available CPU cores

    pi_estimate = parallel_monte_carlo_pi(total_samples, num_processes)
    print(f"Estimated value of π: {pi_estimate}")
