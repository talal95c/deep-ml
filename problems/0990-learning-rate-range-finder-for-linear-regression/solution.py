import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    w = w0.copy()
    losses_hist = []
    n = X.shape[0]
    

    for k in range(n_steps):
        if n_steps == 1 :
            Lr_k =10**a
        else : 
            exposant= (a+((b-a)*k /(n_steps-1)))
            Lr_k = 10**exposant

        prediction = X @ w 
        erreur = prediction - y 
        grad =2/n* X.T @ erreur
        w = w - Lr_k*grad

        prediction = X @ w 
        losses_lr = np.mean((prediction-y)**2)
        losses_hist.append(float(losses_lr))



    return losses_hist
