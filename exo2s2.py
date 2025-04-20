import numpy as np
from scipy import linalg

# Définir la matrice A
A = np.array([
    [4, 2, -2, 6],
    [2, 5, 5, 1],
    [-2, 5, 26, -10],
    [6, 1, -10, 12]
])

# Décomposition de Cholesky manuelle
def cholesky(A):
    n = len(A)
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    raise ValueError("Matrice non définie positive")
                L[i][j] = np.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

# Résolution manuelle
def substitution_avant(L, b):
    n = len(b)
    Y = np.zeros(n)
    for i in range(n):
        s = sum(L[i][j] * Y[j] for j in range(i))
        Y[i] = (b[i] - s) / L[i][i]
    return Y

def substitution_arriere(LT, Y):
    n = len(Y)
    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sum(LT[i][j] * X[j] for j in range(i + 1, n))
        X[i] = (Y[i] - s) / LT[i][i]
    return X

# Exécution
L = cholesky(A)
LT = L.T
b = np.array([1, 0, 0, 0])

# Résolution manuelle
Y_manual = substitution_avant(L, b)
X_manual = substitution_arriere(LT, Y_manual)

# Résolution via scipy
Y_scipy = linalg.solve_triangular(L, b, lower=True)
X_scipy = linalg.solve_triangular(LT, Y_scipy, lower=False)

# Affichage des résultats
print("Solution manuelle de L * Y = b :")
print(Y_manual)

print("Solution scipy de L * Y = b :")
print(Y_scipy)

print("Solution manuelle de L^T * X = Y :")
print(X_manual)

print("Solution scipy de L^T * X = Y :")
print(X_scipy)

