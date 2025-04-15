import matplotlib.pyplot as plt
import numpy as np

#le code affiche la matrice apres chaque iteration ,madame.
def display_augmented_matrix(A, b, title=""):
    print(f"--- {title} ---")
    print("A | b")
    for row, bb in zip(A, b):
        formatted_row = ["{:8.4f}".format(val) for val in row]
        print(formatted_row, "|", "{:8.4f}".format(bb))
    print("------------------------------\n")

def plot_matrix(A, step):
    plt.figure(figsize=(6, 5))
    matrix_array = np.array(A, dtype=float)
    plt.imshow(matrix_array, cmap='coolwarm', interpolation='nearest')

    plt.colorbar(label="Valeur des coefficients")
    plt.title(f"Matrice A - Étape {step} de l’élimination")
    
    for i in range(len(matrix_array)):
        for j in range(len(matrix_array[i])):
            plt.text(j, i, f"{matrix_array[i][j]:.2f}", ha='center', va='center', color='black', fontsize=9)

    plt.xlabel("Colonnes")
    plt.ylabel("Lignes")
    plt.tight_layout()
    plt.show()

def gauss_elimination(A, b):
    n = len(A)
    for k in range(n - 1):
        pivot = A[k][k]
        if abs(pivot) < 1e-14:
            raise ValueError(f"Zero (or very small) pivot encountered at row {k}!")
        
        for i in range(k + 1, n):
            m = A[i][k] / pivot
            for j in range(k, n):
                A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
        
        display_augmented_matrix(A, b, f"Après l’étape {k+1}")
        plot_matrix(A, k + 1)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / A[i][i]
    
    return x

def main():
    A = [
        [ 2, -1,  4,  0],
        [ 4, -1,  5,  1],
        [-2,  2, -2,  3],
        [ 0,  3, -9,  1]
    ]
    b = [1, 0, 0, 0]
    
    print("=== Système initial (matrice augmentée) ===")
    display_augmented_matrix(A, b, "Système initial")

    x = gauss_elimination(A, b)

    print("=== Matrice triangulaire supérieure (final A|b) ===")
    display_augmented_matrix(A, b, "Matrice finale")

    print("=== Solution ===")
    for i, val in enumerate(x, start=1):
        print(f"x{i} = {val:.4f}")

if __name__ == "__main__":
    main()
