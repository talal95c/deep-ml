import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    n = len(predicted_probs)
    summ = 0 
    for i in range(len(predicted_probs)):
        for j in range(len(predicted_probs[i])):
            summ += true_labels[i][j] *np.log(max(epsilon,predicted_probs[i][j]+epsilon))
    for row in predicted_probs:
        if not np.isclose(np.sum(row), 1.0, atol=1e-6):
            return None
    return -1/n * (summ)