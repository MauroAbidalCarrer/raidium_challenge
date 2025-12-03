from time import time
from contextlib import contextmanager


time_dict: dict[str, float] = {}


@contextmanager
def time_to_run(context_name: str):
    start = time()
    try:
        yield
    finally:
        end = time()
        elapsed_time = end - start
        time_dict[context_name] = time_dict.get(context_name, 0) + elapsed_time

def print_time_dict():
    time_sum = 0
    for k, v in time_dict.items():
        print(f"{k}: {v:.4f}s")
        time_sum += v
    print(f"\ntotals to: {time_sum:.2f}")