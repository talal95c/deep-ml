import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):

    np.random.seed(random_seed)

    classes = np.unique(y)

    train_indices = []
    test_indices = []

    for classe in classes:

        indice = np.where(y == classe)[0]

        indice = np.random.permutation(indice)

        n_test = int(len(indice) * test_size)

        test_indices.extend(indice[:n_test])
        train_indices.extend(indice[n_test:])

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test