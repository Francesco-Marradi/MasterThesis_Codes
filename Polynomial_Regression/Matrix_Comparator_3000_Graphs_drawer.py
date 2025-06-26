#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI----------------------------------------------- #
#------------------------------------------- SCRIPT OBJECTIVE --------------------------------------------------------- #

# The objective of this script is to load the dynamical matrices, take the function in each cell and create a dataset from
# it, then, use that data set to approximate the behaviour of the cell's function with a polynomial derived using a 
# polynomial regression technique.

# -------------------------------------------------------------------------------------------------------------------- #




#------------------------------------ IMPORTING THE NECESSARY PACKAGES ----------------------------------------------- #

import sympy as sy
import numpy as np
import sympy.physics.mechanics as me
import os
import json
from functools import reduce
from operator import mul
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor, parse_expr


# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

#Cleaning the terminal SSSS
os.system('cls')

print('231')


# ---------------------------------------- LOADING THE DYNAMICAL MATRICES -------------------------------------------- #
print('Loading dynamical matrices...')

import json
import matplotlib.pyplot as plt
import numpy as np

with open('difference_matrices_Satellite.json', 'r') as f:
    diff_data = json.load(f)

time_values = diff_data['time_values']
COEF_diff_loaded = diff_data['COEF_diff']
RHS_diff_loaded = diff_data['RHS_diff']
COEF_loaded = diff_data['COEF_results']
RHS_loaded = diff_data['RHS_results']
PolyCOEF_loaded = diff_data['PolyCOEF_results']
PolyRHS_loaded = diff_data['PolyRHS_results']


# ------------------------------------------- PLOT CELL ERROR OVER TIME ------------------------------------------------------ #
def plot_error_over_time(matrix_type="COEF", row=0, col=0):
    if matrix_type == "COEF":
        complex_values = COEF_loaded[row][col]
        Poly_values = PolyCOEF_loaded[row][col]
        err = [x * 100 for x in COEF_diff_loaded[row][col]]
        title = f"Error Plot - Mass Matrix cell [{row},{col}] - R² = 0.999"
    elif matrix_type == "RHS":
        complex_values = RHS_loaded[row][col]
        Poly_values = PolyRHS_loaded[row][col]
        err = [x * 100 for x in RHS_diff_loaded[row][col]]
        title = f"Error Plot - Forcing Vector cell [{row},{col}]"
    else:
        raise ValueError("matrix_type deve essere 'COEF' oppure 'RHS'.")


    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primo asse y (a sinistra) per le cell values
    ax1.plot(time_values, complex_values, label="Original expression", linestyle='-', color='#5c8a99')
    # ax1.plot(time_values, Poly_values, label="Polynomial epxression", linestyle='-', color='#6a994e')
    ax1.set_ylabel("Expressions output values")
    ax1.set_xlabel("t [s]")
    ax1.grid(True)
    ax1.set_ylim(0,np.max(complex_values)*1.5)
    ax1.set_xlim(0,10)

    # Secondo asse y (a destra) per gli errori in %
    ax2 = ax1.twinx()
    # ax2.plot(time_values, err, label="Interpolation Error", linestyle='-', color='#ba1200')
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("Interpolation Error value[%]")
    # ax2.set_ylim(0,np.max(err)*2)
    ax2.set_xlim(0,10)

    # Titolo e layout
    plt.title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    fig.tight_layout()
    plt.show()

# ------------------------------------------- FUNZIONI PER TROVARE E PLOTTARE LE TOP 10 CELLE ------------------------------------------------------ #

def get_top_error_cells(matrix_diff, matrix_type="COEF", top_n=10):
    error_list = []
    n_rows = len(matrix_diff)
    n_cols = len(matrix_diff[0])

    for i in range(n_rows):
        for j in range(n_cols):
            total_error = sum(abs(val) for val in matrix_diff[i][j])
            error_list.append(((i, j), total_error))

    error_list.sort(key=lambda x: x[1], reverse=True)

    print(f"🔍 Top {top_n} celle con errore totale maggiore nella matrice {matrix_type}:")
    for rank, ((i, j), err) in enumerate(error_list[:top_n], 1):
        print(f"{rank}. Cella [{i},{j}] -> Errore totale: {err:.4f}")

    return [cell for cell, _ in error_list[:top_n]]


def plot_top_error_cells(matrix_diff, matrix_type="COEF", top_n=10):
    top_cells = get_top_error_cells(matrix_diff, matrix_type, top_n)

    for i, j in top_cells:
        plot_error_over_time(matrix_type, i, j)

# ------------------------------------------- AVVIO ANALISI ------------------------------------------------------ #

# 👇 Usa questi comandi per visualizzare automaticamente le celle più critiche:
plot_top_error_cells(COEF_diff_loaded, matrix_type="COEF", top_n=10)
plot_top_error_cells(RHS_diff_loaded, matrix_type="RHS", top_n=10)



# -------------------------------------------------------------------------------------------------------------------- #
print()
print('Codice terminato')







