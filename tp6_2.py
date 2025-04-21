import numpy as np
import matplotlib.pyplot as plt

A = [
    [1.0, 2.0, 4.0],
    [1.0/8, 1.0, 1.0],
    [-1.0, 4.0, 1.0]
]

b = [1.0, 3.0, 7.0]
n = len(b)
def construire_matrice_jacobi(A):
    J = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i][j] = -A[i][j] / A[i][i]
            else:
                J[i][j] = 0.0
    return J

J_A = construire_matrice_jacobi(A)

print(" Matrice de Jacobi J_A :")
for ligne in J_A:
    print([round(x, 4) for x in ligne])
def produit_matrice_vecteur(M, V):
    resultat = [0.0 for _ in range(len(M))]
    for i in range(len(M)):
        for j in range(len(V)):
            resultat[i] += M[i][j] * V[j]
    return resultat

def norme_inf(V):
    return max(abs(x) for x in V)

def soustraction_vecteurs(v1, v2):
    return [a - b for a, b in zip(v1, v2)]

def jacobi(A, b, X0, iterations):
    n = len(A)
    X = X0.copy()
    historique = [X.copy()]

    for k in range(iterations):
        X_new = [0.0] * n
        for i in range(n):
            somme = 0.0
            for j in range(n):
                if j != i:
                    somme += A[i][j] * X[j]
            X_new[i] = (b[i] - somme) / A[i][i]
        X = X_new
        historique.append(X.copy())
    
    return historique
def puissance_iterative(M, tol=1e-10, max_iter=100):
    n = len(M)
    v = [1.0] * n
    lambda_old = 0.0

    for _ in range(max_iter):
        w = produit_matrice_vecteur(M, v)
        norm_v = norme_inf(w)
        v = [x / norm_v for x in w]
        lambda_new = norme_inf(w)
        if abs(lambda_new - lambda_old) < tol:
            break
        lambda_old = lambda_new

    return lambda_new

A_np = np.array(A)
b_np = np.array(b)
X_exact = np.linalg.solve(A_np, b_np)

print("\n🧮 Solution exacte via numpy.linalg.solve :")
print([round(x, 6) for x in X_exact])

X0 = [0.0, 1.0, 0.0]
max_iter = 15
solutions = jacobi(A, b, X0, max_iter)

print("\n🔁 Itérations de Jacobi :")
for i, vec in enumerate(solutions[:6]): 
    print(f"X^{i} =", [round(x, 6) for x in vec])


erreurs = []
for x in solutions:
    erreur = norme_inf(soustraction_vecteurs(x, X_exact))
    erreurs.append(erreur)

plt.figure(figsize=(8, 5))
plt.plot(range(len(erreurs)), erreurs, marker='o', linestyle='-', color='blue')
plt.title("Convergence de la méthode de Jacobi")
plt.xlabel("Itération k")
plt.ylabel("Erreur ||X(k) - X*||∞")
plt.grid(True)
plt.tight_layout()
plt.show()
rayon_spectral = puissance_iterative(J_A)
print("\n🔍 Rayon spectral approximé de J_A :", round(rayon_spectral, 6))
if rayon_spectral < 1:
    print("Convergence garantie (rayon spectral < 1)")
else:
    print("Pas de convergence (rayon spectral ≥ 1)")
