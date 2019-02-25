from typing import Callable

import numpy as np

from willump import *


def willump_cache(func: Callable, args: tuple, willump_cache_dict, cache_index: int):
    cache_dict_caches = willump_cache_dict[WILLUMP_CACHE_NAME]
    cache_dict_max_lens = willump_cache_dict[WILLUMP_CACHE_MAX_LEN_NAME]
    current_iter_number = willump_cache_dict[WILLUMP_CACHE_ITER_NUMBER]
    willump_cache_dict[WILLUMP_CACHE_ITER_NUMBER] += 1
    cache = cache_dict_caches[cache_index]
    max_len = cache_dict_max_lens[cache_index]
    if args in cache:
        cache.move_to_end(args)
        return cache[args]
    else:
        result = func(*args)
        if len(cache) > max_len:
            cache.popitem()
        cache[args] = result
        return result


# TODO:  Use small_model_output to shorten more_important_vecs before combining in batch case.
def cascade_dense_stacker(more_important_vecs, less_important_vecs, small_model_output):
    output = np.hstack((*more_important_vecs, *less_important_vecs))
    return output


def csr_marshall(csr_matrix):
    indices = csr_matrix.indices
    data = csr_matrix.data
    length, width = csr_matrix.shape
    indptr = csr_matrix.indptr
    return indptr, indices, data, length, width
