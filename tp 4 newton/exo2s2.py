import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import linalg
# def A
A = np.array([
    [ 4, 2,  -2,  6],
    [ 2, 5,  5,  1],
    [-2,  5, 26,  -10],
    [ 6,  1, -10,  12]
])
#cholesky methode
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
L = cholesky(A)
print("Matrice L de Cholesky :")
print(L) 
ls = np.linalg.cholesky(A)

print("Matrice L de Cholesky avec numpy :")
print (ls)
transpo = np.transpose(L)
#solve 
b = np.array([1,0,0,0])
Y = linalg.solve_triangular(L,b, lower=True)
print("Solution Y de L * Y = b :")
print(Y)
X = linalg.solve_triangular(transpo,Y, lower=False)
print("Solution X de L^T * X = Y :")
print(X)

