import numpy as np
import matplotlib.pyplot as plt

# define the function f(x)
def f(x):
    return np.log(x) - x + 2

# 1ere question : tracer le graphe de f
x = np.linspace(0.1, 4, 400)
y = f(x)

plt.plot(x, y, label=r"$f(x) = \ln(x) - x + 2$")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Graphe de f(x)")
plt.legend()
plt.grid()
plt.show()

# 3eme question : algorithme de dichotomie
def dichotomie(f, a, b, precision=0.0001):
    assert f(a) * f(b) < 0, "la condition f(a) * f(b) < 0 n'est pas respected"
    
    cpt = 0
    while (b - a) / 2 > precision:
        c = (a + b) / 2
        if f(c) == 0: # ida c == 0 m3ntha hoi la racine
            return c, cpt
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        cpt += 1
    return (a + b) / 2, cpt


# 4eme question : Calculer les racines avec une precision de 10⁻⁵
root, iterations = dichotomie(f, 1, 3)

# 5eme question :
print(f"Racine: x = {root:.5f} en {iterations} iterations")

# 6eme question :
import pandas as pd

table_data = {
    "Intervalle": ["[1, 3]"],
    "Solution Approchee": [root],
    "Erreur (borne sup - borne inf)": [(3 - 1) / (2 ** iterations)]
}
df = pd.DataFrame(table_data)
print(df)











# 7eme question : Representation graphique avec la solution
# plt.plot(x, y, label=r"$f(x) = \ln(x) - x + 2$")
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(root, color='red', linestyle="--", label=f"Racine x ≈ {root:.5f}")
# plt.scatter([root], [0], color="red", marker="*", label="Solution trouvée")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.title("Graphe de f(x) avec la racine te3ha")
# plt.legend()
# plt.grid()
# plt.show()

# # 8eme question :
# def g(x):
#     return np.exp(x) + (x**2)/2 + x - 1

# root_g, iterations_g = dichotomie(g, -2, 1)
# print(f"Racine pour g(x): x = {root_g:.5f} en {iterations_g} itérations.")

# x_g = np.linspace(-2, 1, 400)
# y_g = g(x_g)

# plt.plot(x_g, y_g, label=r"$g(x) = e^x + \frac{x^2}{2} + x - 1$")
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(root_g, color='red', linestyle="--", label=f"Racine x ≈ {root_g:.5f}")
# plt.scatter([root_g], [0], color="red", marker="*", label="sebna solution")
# plt.xlabel("x")
# plt.ylabel("g(x)")
# plt.title("Graphe de g(x) avec solution trouvée")
# plt.legend()
# plt.grid()
# plt.show()