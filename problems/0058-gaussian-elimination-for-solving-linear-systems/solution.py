import numpy as np

def gaussian_elimination(A, b):
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).reshape(-1).copy()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A doit être une matrice carrée.")

    n = A.shape[0]

    if b.size != n:
        raise ValueError("La taille de b doit correspondre à celle de A.")

    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A et b doivent contenir des valeurs finies.")

    # 1. Pivotement partiel et élimination
    for j in range(n):
        index = j
        max_val = abs(A[j, j])

        # Chercher le plus grand coefficient en valeur absolue
        for i in range(j + 1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                index = i

        if max_val == 0:
            raise ValueError("Le système n'a pas de solution unique.")

        # Échanger les lignes dans A et b
        temp_a = A[j].copy()
        temp_b = b[j]

        A[j] = A[index]
        b[j] = b[index]

        A[index] = temp_a
        b[index] = temp_b

        # Annuler les coefficients sous le pivot
        for i in range(j + 1, n):
            facteur = A[i, j] / A[j, j]

            A[i] -= facteur * A[j]
            b[i] -= facteur * b[j]
            A[i, j] = 0.0

    # 2. Remontée
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        somme = 0.0

        for k in range(i + 1, n):
            somme += A[i, k] * x[k]

        x[i] = (b[i] - somme) / A[i, i]

    return x