import numpy as np
import matplotlib.pyplot as plt

# Déclaration des fonctions
def f1(x):
    return x**6 - x - 1

def f2(x):
    return 1 - (1/4) * np.cos(x)

def f3(x):
    return np.cos(x) - np.exp(-x)

def f4(x):
    return x**4 - 56.101*x**3 + 785.6561*x**2 - 72.7856*x + 0.078


# Création des subplots (2 lignes, 2 colonnes)
plt.figure(figsize=(10,10))
def graphique(Fn , a , b,nom,i,sol):
    x = np.linspace(a, b, 100)
    y = Fn(x)
    plt.subplot(2,2,i)
    plt.plot(x, y, "b-")
    y1=Fn(sol)
    plt.scatter(sol, y1, color='red', label='Point d\'intersection')
    plt.axhline(y=0 , color='black', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.title(f"{nom}")
    plt.legend()

x1=np.array([-0.75,1.124])
x2=np.array([0,0.75])
x3=np.array([0,1.284])
x4= np.linspace(0,4,2)
graphique(f1, -2, 2, "f1",1,x1)

graphique(f2, -10, 10, "f2",2,0)

graphique(f3, 0, 4, "f3",3,x3)

graphique(f4, -10, 10, "f4",4,x2)

plt.tight_layout()
plt.show()