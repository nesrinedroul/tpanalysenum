import numpy as np

# Exemple d'une matrice symétrique définie positive
A = np.array([
    [1, 2, 3, -1],
    [2, -1, 9, -7],
    [-3, 4, -3, 19],
    [4, -2, 6, -21]
], dtype=float)

b = np.array([1, 0, 0, 0], dtype=float)


def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)

def determinant_mineur_principal(matrix, n):
    return np.linalg.det(matrix[:n, :n])

def is_positive_definite(matrix):
    n = matrix.shape[0]
    for i in range(1, n + 1):
        # Calculer le déterminant du mineur principal de taille i
        det = determinant_mineur_principal(matrix, i)
        if det <= 0:
            return False
    return True
def cholesky_manual(matrix):
    n = matrix.shape[0]
    L = np.zeros_like(matrix)

    for i in range(n):
        for j in range(i+1):
            sum_ = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                value = matrix[i][i] - sum_
                if value <= 0:
                    raise ValueError("Matrice non définie positive.")
                L[i][j] = np.sqrt(value)
            else:
                L[i][j] = (matrix[i][j] - sum_) / L[j][j]
    return L

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

# Résolution L^T x = y (remontée)
def backward_substitution(LT, y):
    n = len(y)
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(LT[i, i+1:], x[i+1:])) / LT[i, i]
    return x

# Vérifications
print("Matrice symétrique :", is_symmetric(A))

if is_positive_definite(A):
    print("Matrice définie positive : oui")
    
    try:
        L = cholesky_manual(A)
        print("Matrice L :\n", L)

        y = forward_substitution(L, b)
        x = backward_substitution(L.T, y)

        print("\n🔍 Étape 1 : Résolution de L y = b → y =", y)
        print("🔍 Étape 2 : Résolution de Lᵗ x = y → x =", x)

        # Comparaison avec numpy
        x_np = np.linalg.solve(A, b)
        print("\n Comparaison avec numpy.linalg.solve :\n", x_np)

    except ValueError as e:
        print("❌", e)
else:
    print("La matrice n'est pas définie positive.")
