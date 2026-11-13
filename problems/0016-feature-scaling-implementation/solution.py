import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    standardized_data = data.copy()
    
    m = data.mean(axis=0)
    std = data.std(axis=0)
    standardized_data = (standardized_data-m) / std

    normalized_data = data.copy()
    max = data.max(axis=0)
    min = data.min(axis=0)
    normalized_data = (data-min)/(max-min)


    return standardized_data, normalized_data