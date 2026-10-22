import numpy as np


def gradient_method(X, y, batch_size=1, method="batch"):
    m = X.shape[0]

    match method:
        case "batch":
            yield X, y

        case "stochastic":
            

            for index in range(m):
                yield (
                    X[index:index + 1],
                    y[index:index + 1]
                )

        case "mini_batch":
            if batch_size <= 0:
                raise ValueError("batch_size must be greater than 0.")


            for start in range(0, m, batch_size):
                end = start + batch_size

                yield (
                    X[start:end],
                    y[start:end]
                )

        case _:
            raise ValueError(
                "method must be 'batch', 'stochastic', or 'mini_batch'."
            )


def gradient_descent(
    X,
    y,
    weights,
    learning_rate,
    n_epochs,
    batch_size=1,
    method="batch"
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).copy()

    for _ in range(n_epochs):
        for X_batch, y_batch in gradient_method(
            X,
            y,
            batch_size=batch_size,
            method=method
        ):
            predictions = X_batch @ weights
            errors = predictions - y_batch
            batch_length = X_batch.shape[0]
            gradient = (2 / batch_length) * (X_batch.T @ errors)
            weights = weights - learning_rate * gradient

    return weights