import numpy as np

# Données du système
A = np.array([
    [4, 2, -2, 0],
    [2, 5, 5, 1],
    [-2, 5, 26, -10],
    [6, 1, -10, 12]
], dtype=float)

b = np.array([1, 0, 0, 0], dtype=float)

# Étape 1 : Décomposition manuelle de A en D, E, F
n = A.shape[0]
D = np.zeros_like(A)
E = np.zeros_like(A)
F = np.zeros_like(A)

for i in range(n):
    for j in range(n):
        if i == j:
            D[i, j] = A[i, j]
        elif i > j:
            E[i, j] = -A[i, j]  # Signe négatif pour Jacobi
        else:
            F[i, j] = -A[i, j]  # Signe négatif pour Jacobi

# Étape 2 : Construction de la matrice de Jacobi T et du vecteur c
D_inv = np.linalg.inv(D)
T = D_inv @ (E + F)
c = D_inv @ b

# Rayon spectral
rho = max(abs(np.linalg.eigvals(T)))
print(" Matrice T (Jacobi) =\n", T)
print(" Vecteur c =\n", c)
print(" Rayon spectral rho(T) =", rho)
if rho < 1:
    print(" La méthode de Jacobi converge\n")
else:
    print("La méthode de Jacobi ne converge pas\n")

# Étape 3 : Itérations de Jacobi
def jacobi_verbose(A, b, x0, tol=1e-6, max_iter=100):
    n = A.shape[0]
    D = np.zeros_like(A)
    E = np.zeros_like(A)
    F = np.zeros_like(A)

    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = A[i, j]
            elif i > j:
                E[i, j] = -A[i, j]
            else:
                F[i, j] = -A[i, j]

    D_inv = np.linalg.inv(D)
    T = D_inv @ (E + F)
    c = D_inv @ b

    x = x0.copy()

    print(" Début des itérations Jacobi :\n")
    for k in range(max_iter):
        x_new = T @ x + c
        print(f"🔷 Itération {k+1}:")
        print("x^(k) =", x)
        print("x^(k+1) =", x_new)
        print("Norme erreur (infini):", np.linalg.norm(x_new - x, ord=np.inf))
        print("-" * 40)

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"\nConvergence atteinte après {k+1} itérations.")
            return x_new

        x = x_new

    print("\nConvergence non atteinte après", max_iter, "itérations.")
    return x

# Valeur initiale
x0 = np.zeros_like(b)

# Exécution
solution = jacobi_verbose(A, b, x0)
print("\n Solution approchée finale x =", solution)
