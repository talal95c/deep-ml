import numpy as np

def norme2(x:int):
    return np.sqrt(np.sum(x**2))


def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    # Your code here
    x = np.asarray(x, dtype=float)
    analytical_grad = np.asarray(analytical_grad, dtype=float)
    numerical_grad= np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_moins= x.copy()

        x_plus[i] += epsilon
        x_moins[i] -= epsilon

        numerical_grad[i] = (
           ( f(x_plus )-f(x_moins))/(2*epsilon)
        )


    denominateur = norme2(numerical_grad)+norme2(analytical_grad)
    if denominateur == 0:
        relative_error = 0.0
    else : 
        relative_error = (
            norme2(numerical_grad-analytical_grad)/denominateur
        )


    return(numerical_grad,relative_error)
