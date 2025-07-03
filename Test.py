import time

def simulate_long_task(iterations):
    """
    Simulates a computational task by performing a large number of additions.
    """
    start_time = time.time() # Record the start time

    result = 0
    for i in range(iterations):
        result += i * 2 # Perform a simple operation

    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time # Calculate elapsed time

    print(f"Task completed with {iterations} iterations.")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return result

# --- How to use it ---
# Adjust the 'num_iterations' to control how long it takes.
# For a few seconds, start with something like 100,000,000 (100 million)
# This will vary *greatly* depending on your CPU speed.

num_iterations = 100_000_000 # Try 100 million for a few seconds on a modern CPU

print(f"Starting a task with {num_iterations} iterations...")
simulate_long_task(num_iterations)

# To make it take longer, increase num_iterations, e.g., to 500_000_000 or 1_000_000_000
# print("\nStarting a longer task...")
# simulate_long_task(500_000_000)