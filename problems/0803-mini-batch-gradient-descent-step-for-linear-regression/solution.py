import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X = np.asarray(X,dtype=float)
    y = np.asarray(y,dtype=float).reshape(-1)
    w = np.asarray(weights,dtype=float).copy()
    b = np.asarray(bias,dtype=float).copy()


    m = len(batch_indices) 
    X_batches = X[batch_indices]
    y_batches = y[batch_indices]


    prediciton = X_batches @ w + b 
    error = prediciton - y_batches
    grad_w = (2/m) * X_batches.T @ error
    grad_b = (2/m) * np.sum(error)  
    w = w - grad_w*lr
    b= b -grad_b*lr

    return np.concatenate((w,np.array([b])))

    
