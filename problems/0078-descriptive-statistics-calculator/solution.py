import numpy as np
from collections import Counter

def descriptive_statistics(data: list | np.ndarray) -> dict:
    n = len(data)

    mean = (1 / n) * sum(data)

    variance = (1 / n) * sum((x - mean) ** 2 for x in data)

    sorted_list = sorted(data)

    if n % 2 == 0:
        median = (sorted_list[n // 2 - 1] + sorted_list[n // 2]) / 2
    else:
        median = sorted_list[n // 2]

    mode = Counter(data).most_common(1)[0][0]

    standard_deviation = np.sqrt(variance)

    percentile_25 = np.percentile(data, 25)
    percentile_50 = np.percentile(data, 50)
    percentile_75 = np.percentile(data, 75)

    interquartile_range = percentile_75 - percentile_25

    return {
        'mean': mean,
        'median': median,
        'mode': mode,
        'variance': variance,
        'standard_deviation': standard_deviation,
        '25th_percentile': percentile_25,
        '50th_percentile': percentile_50,
        '75th_percentile': percentile_75,
        'interquartile_range': interquartile_range,
    }