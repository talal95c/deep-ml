import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    data_shuffled = data[permutation]

    train_end = int(n*train_frac)
    validation_end = train_end + int(n*validation_frac)

    train_data = data_shuffled[:train_end]
    validation = data_shuffled[train_end:validation_end]
    test = data_shuffled[validation_end:]

    return train_data,validation,test



    pass