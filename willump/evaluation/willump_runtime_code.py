from typing import Callable

import numpy as np
import pandas as pd

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


def cascade_dense_stacker(more_important_vecs, less_important_vecs, small_model_output):
    if not isinstance(more_important_vecs[0], list):
        indices = [i for i in range(len(small_model_output)) if small_model_output[i] == 2]
        for i, entry in enumerate(more_important_vecs):
            more_important_vecs[i] = entry[indices]
    output = np.hstack((*more_important_vecs, *less_important_vecs))
    return output


def cascade_df_shorten(df, small_model_output):
    indices = [i for i in range(len(small_model_output)) if small_model_output[i] == 2]
    if isinstance(df, list):
        return [df[i] for i in indices]
    elif isinstance(df, pd.DataFrame):
        return df.iloc[indices]
    else:
        return df[indices]


def csr_marshall(csr_matrix):
    indices = csr_matrix.indices
    data = csr_matrix.data
    length, width = csr_matrix.shape
    indptr = csr_matrix.indptr
    return indptr, indices, data, length, width
