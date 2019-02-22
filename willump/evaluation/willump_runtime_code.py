from typing import Callable
import numpy as np


def willump_cache(func: Callable, args: tuple, cache: dict):
    if args in cache:
        return cache[args]
    else:
        result = func(*args)
        cache[args] = result
        return result


def cascade_dense_stacker(more_important_vecs, less_important_vecs, small_model_output):
    output = np.hstack((*more_important_vecs, *less_important_vecs))
    return output
