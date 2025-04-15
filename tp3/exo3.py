import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x - 1 - np.exp(-x)

def phi1(x):
    return np.exp(-x) + 1  

def dphi1(x):
    return -np.exp(-x)  
x_vals = np.linspace(1, 2, 100)
y_vals = f(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label=r"$f(x) = x - 1 - e^{-x}$", color='b')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Vérification graphique de la racine de f(x)")
plt.legend()
plt.grid()
plt.show()

k = max(abs(dphi1(x_vals)))
print(f"Valeur de k sur [1,2]: {k}")

def point_fixe(phi, x0, k, epsilon=1e-6, max_iterations=100):
    x = x0
    erreurs_thero = []  
    solutions_approchees = [x]  

    for i in range(max_iterations):
        xi = phi(x)
        error = (k**i / (1 - k)) * abs(xi - x) 
        erreurs_thero.append(error)
        solutions_approchees.append(xi)
        if abs(xi - x) < epsilon:
            return xi, erreurs_thero, solutions_approchees  
        x = xi  

    return None, erreurs_thero, solutions_approchees  
x0 = 1 
solution, erreurs_thero, solutions_approchees = point_fixe(phi1, x0, k)
if solution is not None:
    print(f"Solution approximée trouvée : {solution}")
else:
    print("Pas de convergence avec la méthode du point fixe.")

print("\nItération | Solution Approchée | Erreur Théorique")
for i in range(len(erreurs_thero)):
    print(f"{i+1:<9} | {solutions_approchees[i]:<18} | {erreurs_thero[i]:<18}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(erreurs_thero) + 1), erreurs_thero, marker='o', linestyle='-', color='r', label="Erreur théorique")
plt.xlabel("Itérations")
plt.ylabel("Erreur théorique")
plt.title("Évolution de l'erreur théorique au fil des itérations")
plt.yscale("log")  
plt.legend()
plt.grid()
plt.show()
