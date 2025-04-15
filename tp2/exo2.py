import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.log(x) - x + 2

# Function to plot f(x) with multiple solutions
def graphique(Fn, a, b, sol):
    x = np.linspace(a, b, 400)
    y = Fn(x)
    plt.plot(x, y, "b-", label="f(x)")
    plt.scatter(sol, Fn(sol), color='red', label="Points d'intersection")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.title("f(x)")
    plt.legend()
    plt.show()

sol = np.array([0.158, 3.142])
graphique(f, 0.1, 5, sol)

# Dichotomy method
def dichotomie(f, a, b, precision=0.00001):
    if f(a) * f(b) > 0:
        print("La condition f(a) * f(b) < 0 est violée")
        return None, None
    
    iteration = 0
    while (b - a) / 2 > precision:
        c = (a + b) / 2
        if f(c) == 0:
            return c, iteration
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iteration += 1
        return (a + b) / 2, iteration

racine, iteration = dichotomie(f, 2.5, 3.25)
racine2, iteration2 = dichotomie(f, 0.12, 0.25)

print(f"Racine: x = {racine:.5f} en {iteration} iterations")
print(f"Racine 2: x = {racine2:.5f} en {iteration2} iterations")
def graphiquesol(Fn, a, b, roots):
    x = np.linspace(a, b, 400)
    y = Fn(x)
    plt.plot(x, y, "b-", label="f(x)")
    roots = np.array(roots)
    plt.scatter(roots, np.zeros_like(roots), color="blue", marker="*", label="Solutions trouvées")
    plt.axvline(roots[0], color='red', linestyle="--", label=f"Racine x ≈ {roots[0]:.5f}")
    plt.axvline(roots[1], color='blue', linestyle="--", label=f"Racine x ≈ {roots[1]:.5f}")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.title("f(x) avec Racines")
    plt.legend()
    plt.show()

roots = np.array([racine, racine2])
graphiquesol(f, 0.1, 5, roots)

print(f"{'Intervalle':<15} {'Solution Approchée':<20} {'Erreur (borne sup - borne inf)':<20}")
print("-" * 60)
#b-a/(2**iterations)
print(f"[2.5, 3.25]     {racine:.5f}              {(3.25 - 2.5) / (2 ** iteration):.5f}")
print(f"[0.12, 0.25]    {racine2:.5f}              {(0.25 - 0.12) / (2 ** iteration2):.5f}")
def y(x): 
    return np.exp(x) + (x**2)/2 + x - 1


