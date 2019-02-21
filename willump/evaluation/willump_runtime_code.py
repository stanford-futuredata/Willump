from typing import Callable


def willump_cache(func: Callable, args: tuple, cache: dict):
    if args in cache:
        return cache[args]
    else:
        result = func(*args)
        cache[args] = result
        return result
