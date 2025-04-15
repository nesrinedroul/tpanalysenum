import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(x) + x**2 / 2 + x - 1

def f_prime(x):
    return np.exp(x) + x + 1

def f_double_prime(x):
    return np.exp(x) + 1

def newton_method(f, f_prime, f_double_prime, x0, tol, M, m):
    xn = x0
    history = [(0, xn, 0)]
    
    for n in range(1, 100):
        fx = f(xn)
        fpx = f_prime(xn)
        if fpx == 0:
            raise ZeroDivisionError("Dérivée nulle, impossible de continuer.")

        x_next = xn - fx / fpx
        erreur = (M / (2 * m)) * abs(x_next - xn)**2

        history.append((n, x_next, erreur))

        if erreur < tol:
            break
        xn = x_next

    return x_next, history

# Dichotomie
def dichotomie(f, a, b, tol=1e-5, max_iter=100):
    history = []
    for n in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        history.append((n, c, abs(b - a)))
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, history
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, history

# Calcul des bornes m et M
x_check = np.linspace(-2, 2, 1000)
M = np.max(f_double_prime(x_check))
m = np.min(np.abs(f_prime(x_check)))

print(f"M = {M:.5f}, m = {m:.5f}, M/2m = {M/(2*m):.5f}")

tol = 1e-5
sol_newton, hist_newton = newton_method(f, f_prime, f_double_prime, x0=-1, tol=tol, M=M, m=m)
sol_dicho, hist_dicho = dichotomie(f, -2, 1) 

x_vals = np.linspace(-2, 2, 400)
y_vals = f(x_vals)

plt.figure(figsize=(14, 6))

# dichotomie 
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, label='f(x)', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.title("Méthode de dichotomie")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.scatter([x for _, x, _ in hist_dicho], [f(x) for _, x, _ in hist_dicho],
            color='orange', label='Approximations')
plt.scatter(sol_dicho, f(sol_dicho), color='green', marker='x', s=100, label='Solution approx.')
plt.legend()
#newton
plt.subplot(1, 2, 2)
plt.plot(x_vals, y_vals, label='f(x)', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.title("Méthode de Newton (x0 = -1) avec tangentes")
plt.xlabel("x")
plt.ylabel("f(x)")

# Affichage des tangentes
for n, x_n, _ in hist_newton[:-1]: 
    y_n = f(x_n)
    slope = f_prime(x_n)
    x_line = np.linspace(x_n - 1, x_n + 1, 100)
    y_line = slope * (x_line - x_n) + y_n
    plt.plot(x_line, y_line, linestyle='--', color='orange', alpha=0.6)
    plt.scatter(x_n, y_n, color='red')
    plt.text(x_n, y_n, f"x{n}", fontsize=9)

plt.scatter(sol_newton, f(sol_newton), color='green', marker='x', s=100, label='Solution approx.')
plt.legend()

plt.tight_layout()
plt.show()
