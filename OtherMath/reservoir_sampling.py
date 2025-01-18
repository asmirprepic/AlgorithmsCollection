import random

def reservoir_sampling(stream, k):
    """
    Selects k items randomly from a stream of unknown size.

    Args:
        stream: An iterable representing the stream.
        k: Number of items to select.

    Returns:
        A list containing k randomly selected items.
    """
    # Initialize an empty reservoir
    reservoir = []
    
    # Fill the reservoir with the first k items
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Generate a random index
            j = random.randint(0, i)
            # If the index is within the reservoir, replace it
            if j < k:
                reservoir[j] = item
    
    return reservoir
