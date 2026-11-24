import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    def sigmoid(x:np.ndarray):
        return 1/(1+np.exp(-x))


    X = features 
    n = len(X)
    y = labels 
    w = initial_weights
    b = initial_bias
    lr= learning_rate 
    epochs = epochs
    mse_values = []


    for epoch in range(epochs): 
        prediction = X @ w + b
        pred_act = sigmoid(prediction)

        error = pred_act - y 
        mse_values.append(np.mean((error)**2))

        delta = 2 * error * pred_act * (1 - pred_act)
        gradient_w = (X.T @ delta) / n
        gradient_b = np.mean(delta)


        w = w - lr * gradient_w
        b = b - lr * gradient_b

    updated_weights = w
    updated_bias = b  
    return updated_weights, updated_bias, mse_values