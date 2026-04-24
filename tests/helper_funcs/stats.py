'''
stats.py: Statistics helpers for the llama.cpp load tester.

Provides percentile() for computing arbitrary percentiles and
print_stats() for printing a summary of a latency sample.
'''

import statistics


def percentile(data: list[float], p: float) -> float:
    '''Return the p-th percentile of data (0-100).'''

    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1

    if hi >= len(data):
        return data[lo]

    frac = idx - lo
    return data[lo] + frac * (data[hi] - data[lo])


def print_stats(label: str, values: list[float], unit: str = 's') -> None:
    '''Print min, mean, median, p95, and max for a list of float measurements.'''

    if not values:
        print(f'  {label}: no data')
        return

    print(
        f'  {label}: '
        f'min={min(values):.3f}{unit}  '
        f'mean={statistics.mean(values):.3f}{unit}  '
        f'median={statistics.median(values):.3f}{unit}  '
        f'p95={percentile(values, 95):.3f}{unit}  '
        f'max={max(values):.3f}{unit}'
    )
