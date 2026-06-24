# import libraries
import hashlib
import numpy as np
from scipy.stats import gamma, norm

# Convert initial arrays to lists of rows (so appends and removals are cheap)
def to_row_lists(arr):
    return [arr[i] for i in range(arr.shape[0])]

def get_tier_prices(pipe_sizes, price_dict, rng):
    """
    pipe_sizes: array-like of length n
    price_dict: dictionary with keys: 'tiers', 'probs', 'cdf' for each pipe size
    rng: projection specific random number generator
        
    Returns an n x 4 numpy array of sampled tier prices in log space.
    """
    # Initialize result array
    n = len(pipe_sizes)
    result = np.zeros((n, 4))

    # loop over unique pipe sizes
    for ps in np.unique(pipe_sizes):
        # get mask for rows with this pipe size
        mask = (pipe_sizes == ps).reshape(-1)

        # get related price data for this pipe size
        entry = price_dict[ps]
        cdf = entry["cdf"]
        tiers = entry["tiers"]

        # Random draws for all rows with this pipe size
        draws = rng.random(mask.sum())

        # Each draw corresponds to the first index where cdf >= draw
        idxs = np.searchsorted(cdf, draws, side="right")
        
        # Vectorized gather of tier prices
        result[mask] = tiers[idxs]

    return np.log(result)

def make_seed(*args, max_seed=2**32 - 1):
    """
    Create a deterministic integer seed from arbitrary inputs.
    """
    s = "_".join(str(a) for a in args)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max_seed

def make_global_id(id_vec):
    return f"D{id_vec[0]}_C{id_vec[1]}_P{id_vec[2]}_I{id_vec[3]}"

def make_global_id_V2(id_vec):
    return f"D{id_vec[0]}_C{id_vec[1]}_P{id_vec[2]}_I{id_vec[3]}_A{id_vec[4]}"

def make_global_id_HH(id_vec):
    return f"D{id_vec[0]}_C{id_vec[1]}_I{id_vec[2]}_A{id_vec[3]}_P{id_vec[4]}"

def split_vector_into_12(vec, rng):
    """
    Given a vector of integers, split each element into 12 integers that sum to it.
    The remainder is distributed randomly across the 12.
    rng is a projection specific random number generator
    """

    base = np.int32(vec // 12)
    rem = np.int32(vec % 12)
    out = np.tile(base, (12,1)).T  # shape 25x12
    for i, r in enumerate(rem):
        if r>0:
            # remainder is guaranteed to be <12 so no replacement needed
            chosen = rng.choice(12, size=r, replace=False)
            out[i, chosen] += 1
    
    return np.int32(out.flatten())

# Fast swap-pop removal (keeps no particular order, O(1))
def swap_remove(lst, idx):
    # Replace idx with last element and pop last
    last_idx = len(lst) - 1
    if idx != last_idx:
        lst[idx] = lst[last_idx]
    lst.pop()

# Convert lists back to numpy matrix when you need to operate vectorized
def list_to_matrix(lst):
    if len(lst) == 0:
        return np.zeros((0, 0))
    return np.vstack(lst)

def weighted_percentiles(values, weights, percentiles):
    """
    Compute weighted percentiles.

    values : 1-D array
    weights : 1-D array
    percentiles : scalar or 1-D array in [0, 100]

    Returns:
        scalar or array of weighted percentiles
    """

    # sort once
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    # cumulative distribution
    cumulative_weights = np.cumsum(weights)
    cumulative_weights /= cumulative_weights[-1]

    # convert percentiles to [0,1]
    p = percentiles / 100.0

    # interpolation gives smooth results
    return np.interp(p, cumulative_weights, values)