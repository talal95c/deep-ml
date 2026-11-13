import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    #X_train on va l appliquer le fit 

    mnXtrain = X_train.mean(axis=0) # shape : (1,X_train.shape[1])
    stdXtrain= X_train.std(axis=0)
    stdXtrain[stdXtrain == 0] = 1.0
    X_standard = X_test.copy()

    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            X_standard[i,j] = (X_standard[i,j] - mnXtrain[j] )/stdXtrain[j]

    return X_standard
