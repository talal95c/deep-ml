import numpy as np
def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    

    X = np.asarray(X,dtype=float)
    y = np.asarray(y,dtype=float).reshape(-1)
    weights = np.asarray(weights,dtype=float).copy()
    n_samples =X.shape[0]

    for i in range(n_iter):
        index = i % n_samples
        X_index = X[index:index+1]
        y_index = y[index:index+1]

        prediction = X_index @ weights 
        errors = prediction - y_index
        grad = 2*X_index.T @ errors
        weights = weights - learning_rate*grad

    return weights.tolist()

    pass
