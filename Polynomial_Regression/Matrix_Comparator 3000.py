# ---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI----------------------------------------------- #
# ------------------------------------------- SCRIPT OBJECTIVE --------------------------------------------------------- #

# The objective of this script is to load the dynamical matrices, take the function in each cell and create a dataset from
# it, then, use that data set to approximate the behaviour of the cell's function with a polynomial derived using a
# polynomial regression technique.

# -------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------ IMPORTING THE NECESSARY PACKAGES ----------------------------------------------- #

import sympy as sy
import numpy as np
import sympy.physics.mechanics as me
import os
import json
from functools import reduce
from operator import mul
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor, parse_expr


import sys
sys.setrecursionlimit(100000000)

# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

# Cleaning the terminal
os.system('cls')

print('543')

# ---------------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------- #

start = time.perf_counter()

# -------------------------------------------------------------------------------------------------------------------- #


# ---------------------------------------- LOADING THE DYNAMICAL MATRICES -------------------------------------------- #
print('Loading dynamical matrices...')


# Reading the Json file
with open('Matrix_Kane/Clearspace_Satellite/Dynamical_Matrices_4arms_Preloadedsprings_dampers_FINAL.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)

# Reading the Json file
with open('Dynamical_Matrices_4arms_Preloadedsprings_dampers_PolyInterp_R0999.json', 'r') as json_file:
    data_PolyMat_Dyn = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']['COEF_Dyn']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']['RHS_Dyn']))
PolyCOEF_Dyn_loaded = sy.Matrix(sy.sympify(
    data_PolyMat_Dyn['COEF_Dyn']['COEF_Dyn_poly']))
PolyRHS_Dyn_loaded = sy.Matrix(sy.sympify(
    data_PolyMat_Dyn['RHS_Dyn']['RHS_Dyn_poly']))


# ------------------------------------------ SYMBOLIC VARIABLES ----------------------------------------------------#
print('Defining symbols...')


# Initial motion parameters
t = sy.symbols('t')                 # Definition of the time variable
l = sy.symbols('l')                 # Spacecraft bus length [m]
w = sy.symbols('w')                 # Spacecraft bus width [m]
h = sy.symbols('h')                 # Spacecraft bus height [m]
d = sy.symbols('d')                 # Spacecraft bus-arm joint distance [m]
dg = sy.symbols('dg')               # Spacecraft bus joint distance [m]
bl = sy.symbols('bl')               # Robotic arm long piece length [m]
hl = sy.symbols('hl')               # Robotic arm long piece height [m]
wl = sy.symbols('wl')               # Robotic arm long piece width [m]
bs = sy.symbols('bs')               # Robotic arm short piece length [m]
hs = sy.symbols('hs')               # Robotic arm short piece height [m]
ws = sy.symbols('ws')               # Robotic arm short piece width [m]
mlongarm = sy.symbols('mlongarm')   # Robotic Long arm piece mass [kg]
mshortarm = sy.symbols('mshortarm')  # Robotic Short arm piece mass [kg]
mbus = sy.symbols('mbus')           # Spacecraft bus mass [kg]


# Forces and Torques applied
T11 = sy.symbols('T11')             # Torque acting on the arms [N*m]
T21 = sy.symbols('T21')             # Torque acting on the arms [N*m]
T31 = sy.symbols('T31')             # Torque acting on the arms [N*m]
T41 = sy.symbols('T41')             # Torque acting on the arms [N*m]
T12 = sy.symbols('T12')             # Torque acting on the arms [N*m]
T22 = sy.symbols('T22')             # Torque acting on the arms [N*m]
T32 = sy.symbols('T32')             # Torque acting on the arms [N*m]
T42 = sy.symbols('T42')             # Torque acting on the arms [N*m]
T13 = sy.symbols('T13')             # Torque acting on the arms [N*m]
T23 = sy.symbols('T23')             # Torque acting on the arms [N*m]
T33 = sy.symbols('T33')             # Torque acting on the arms [N*m]
T43 = sy.symbols('T43')             # Torque acting on the arms [N*m]
T14 = sy.symbols('T14')             # Torque acting on the arms [N*m]
T24 = sy.symbols('T24')             # Torque acting on the arms [N*m]
T34 = sy.symbols('T34')             # Torque acting on the arms [N*m]
T44 = sy.symbols('T44')             # Torque acting on the arms [N*m]
F = sy.symbols('F')                 # Force acting on the arms [N*m]


# Forces and Torques constants
k11 = sy.symbols('k11')             # B11 body spring's stiffness [N*m/rad]
k21 = sy.symbols('k21')             # B21 body spring's stiffness [N*m/rad]
k31 = sy.symbols('k31')             # B31 body spring's stiffness [N*m/rad]
k41 = sy.symbols('k41')             # B41 body spring's stiffness [N*m/rad]
k12 = sy.symbols('k12')             # B12 body spring's stiffness [N*m/rad]
k22 = sy.symbols('k22')             # B22 body spring's stiffness [N*m/rad]
k32 = sy.symbols('k32')             # B32 body spring's stiffness [N*m/rad]
k42 = sy.symbols('k42')             # B42 body spring's stiffness [N*m/rad]
k13 = sy.symbols('k13')             # B13 body spring's stiffness [N*m/rad]
k23 = sy.symbols('k23')             # B23 body spring's stiffness [N*m/rad]
k33 = sy.symbols('k33')             # B33 body spring's stiffness [N*m/rad]
k43 = sy.symbols('k43')             # B43 body spring's stiffness [N*m/rad]
k14 = sy.symbols('k14')             # B14 body spring's stiffness [N*m/rad]
k24 = sy.symbols('k24')             # B24 body spring's stiffness [N*m/rad]
k34 = sy.symbols('k34')             # B34 body spring's stiffness [N*m/rad]
k44 = sy.symbols('k44')             # B44 body spring's stiffness [N*m/rad]

# B11 body damping coefficient [(N*m/s)/(rad/s)]
c11 = sy.symbols('c11')
# B21 body damping coefficient [(N*m/s)/(rad/s)]
c21 = sy.symbols('c21')
# B31 body damping coefficient [(N*m/s)/(rad/s)]
c31 = sy.symbols('c31')
# B41 body damping coefficient [(N*m/s)/(rad/s)]
c41 = sy.symbols('c41')
# B12 body damping coefficient [(N*m/s)/(rad/s)]
c12 = sy.symbols('c12')
# B22 body damping coefficient [(N*m/s)/(rad/s)]
c22 = sy.symbols('c22')
# B32 body damping coefficient [(N*m/s)/(rad/s)]
c32 = sy.symbols('c32')
# B42 body damping coefficient [(N*m/s)/(rad/s)]
c42 = sy.symbols('c42')
# B13 body damping coefficient [(N*m/s)/(rad/s)]
c13 = sy.symbols('c13')
# B23 body damping coefficient [(N*m/s)/(rad/s)]
c23 = sy.symbols('c23')
# B33 body damping coefficient [(N*m/s)/(rad/s)]
c33 = sy.symbols('c33')
# B43 body damping coefficient [(N*m/s)/(rad/s)]
c43 = sy.symbols('c43')
# B14 body damping coefficient [(N*m/s)/(rad/s)]
c14 = sy.symbols('c14')
# B24 body damping coefficient [(N*m/s)/(rad/s)]
c24 = sy.symbols('c24')
# B34 body damping coefficient [(N*m/s)/(rad/s)]
c34 = sy.symbols('c34')
# B44 body damping coefficient [(N*m/s)/(rad/s)]
c44 = sy.symbols('c44')


#  ---------------------------------------------- #
# Initializing the gen.coordinates and speed
q1, q2, q3, q4, q7, q10, q13, q18, q21, q24, q27, q28, q31, q34, q37, q42, q45, q48, q51, q52, q53, q54 = sy.symbols(
    'q1 q2 q3 q4 q7 q10 q13 q18 q21 q24 q27 q28 q31 q34 q37 q42 q45 q48 q51 q52 q53 q54')
u1, u2, u3, u4, u7, u10, u13, u18, u21, u24, u27, u28, u31, u34, u37, u42, u45, u48, u51, u52, u53, u54 = sy.symbols(
    'u1 u2 u3 u4 u7 u10 u13 u18 u21 u24 u27 u28 u31 u34 u37 u42 u45 u48 u51 u52 u53 u54')

# Putting them into list to manage them better later
qi_list = [q1, q2, q3, q4, q7, q10, q13, q18, q21, q24, q27,
           q28, q31, q34, q37, q42, q45, q48, q51, q52, q53, q54]
ui_list = [u1, u2, u3, u4, u7, u10, u13, u18, u21, u24, u27,
           u28, u31, u34, u37, u42, u45, u48, u51, u52, u53, u54]


# ------------------------------------------------ SUBSTITUTING CONSTANTS -------------------------------------------------- #
print('Substituting constants...')

# Now we have to create a dictionary for the symbolic variables substituition inside the matrices. Practically we need
# to specify the masses, lenghts, ..., values for the system parameters

# Dictionary
substitution = {
    l: 0.6,                 # Spacecraft bus length [m]
    w: 0.5,                 # Spacecraft bus width [m]
    h: 0.5,                 # Spacecraft bus height [m]
    d: 0.1,                 # Spacecraft bus-arm joint distance [m]
    dg: 0.025,              # Spacecraft bus joint distance [m]
    bl: 0.25,               # Robotic arm long piece length [m]
    hl: 0.05,               # Robotic arm long piece height [m]
    wl: 0.05,               # Robotic arm long piece width [m]
    bs: 0.16,               # Robotic arm short piece length [m]
    hs: 0.05,               # Robotic arm short piece height [m]
    ws: 0.05,               # Robotic arm short piece width [m]
    F: 0.0,                 # Force acting on the arms [N]
    mlongarm: 0.836,        # Robotic Long arm piece mass [kg]
    mshortarm: 0.540,       # Robotic Short arm piece mass [kg]
    mbus: 122.8,            # Spacecraft bus mass [kg]

    # Setting to zero the torques
    T11: 0,
    T21: 0,
    T31: 0,
    T41: 0,
    T12: 0,
    T22: 0,
    T32: 0,
    T42: 0,
    T13: 0,
    T23: 0,
    T33: 0,
    T43: 0,
    T14: 0,
    T24: 0,
    T34: 0,
    T44: 0,


    # Defining the spring and damper coefficients
    k11: 0.52,
    k21: 0.3,
    k31: 0.2,
    k41: 0.1,
    k12: 0.52,
    k22: 0.3,
    k32: 0.2,
    k42: 0.1,
    k13: 0.52,
    k23: 0.3,
    k33: 0.2,
    k43: 0.1,
    k14: 0.52,
    k24: 0.3,
    k34: 0.2,
    k44: 0.1,


    c11: 1.2,
    c21: 0.8,
    c31: 0.6,
    c41: 0.4,
    c12: 1.2,
    c22: 0.8,
    c32: 0.6,
    c42: 0.4,
    c13: 1.2,
    c23: 0.8,
    c33: 0.6,
    c43: 0.4,
    c14: 1.2,
    c24: 0.8,
    c34: 0.6,
    c44: 0.4,


}


# Substituing the symbolic variables with their values inside the matrices
# Dynamical Matrices
COEF_Dyn_loaded = COEF_Dyn_loaded.xreplace(substitution)
RHS_Dyn_loaded = RHS_Dyn_loaded.xreplace(substitution)


# ------------------------------------------- GENERATING DATA SETS TO EVALUATE EACH CELL FUNCTION ---------------------------------- #
print('Generating data sets...')

# (Impostazioni per la generazione del dataset)

# First, we need to specify the initial conditions for evaluating it
# u1,  u2,  u3,  u4, u7,  u10, u13, u18, u21  u24  u27  u28  u31  u34  u37  u42  u45  u48  u51  u52  u53  u54  q1   q2   q3   q4   q7   q10  q13  q18  q21  q24  q27  q28  q31  q34  q37  q42  q45  q48  q51  q52  q53  q54
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# 0    1    2   3     4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26  27   28   29   30    31   32   33   34   35   36   37   38   39   40   41   42   43

# ------------------------------------------------------------------------------------------ #
# Costruzione dei valori per le coordinate e velocità generiche per ogni funzione di cella

t_span = 10                 # Definizione del tempo per la generazione dei dati
t_nval = 1000               # Numero di valori per ogni variabile
t_step = t_span / t_nval    # Calcolo dei passi temporali

time_values = np.linspace(0, t_span, t_nval)  # Vettore del tempo

# Inizializzazione della matrice del dataset
Data_set_mat = np.zeros((len(ui_list) + len(qi_list), t_nval))

# Definizione del numero di righe
row_number = len(y0) // 2

# Creazione del ciclo per calcolare i valori del dataset
for i in range(t_nval):
    for j in range(row_number):

        if i == 0:
            # Assegnazione delle condizioni iniziali
            Data_set_mat[:row_number, i] = y0[:row_number]
            # Assegnazione delle condizioni iniziali
            Data_set_mat[row_number:, i] = y0[row_number:]
            continue

        # Calcolo delle coordinate e velocità
        Data_set_mat[j + row_number, i] = (np.random.uniform(-np.pi, np.pi)) * \
            t_step + Data_set_mat[j + row_number, i - 1]
        Data_set_mat[j, i] = (
            Data_set_mat[j + row_number, i] - Data_set_mat[j + row_number, i - 1]) / t_step

        # Eccezioni per le velocità lineari del corpo radice
        if j in (row_number - 3, row_number - 2, row_number - 1):
            Data_set_mat[j + row_number, i] = (
                np.random.uniform(-1, 1)) * t_step + Data_set_mat[j + row_number, i - 1]
            Data_set_mat[j, i] = (
                Data_set_mat[j + row_number, i] - Data_set_mat[j + row_number, i - 1]) / t_step


# ------------------------------------------- EVALUATING EACH CELL FUNCTION ------------------------------------------------------ #
print('Evaluating each cell Function with the dataset...')


def fun_evaluation(fun_to_eval):
    results = np.zeros(t_nval)
    for i in range(t_nval):
        # Estrazione dei dati dal dataset
        inputs = [Data_set_mat[k, i] for k in range(Data_set_mat.shape[0])]
        # Valutazione della funzione
        results[i] = fun_to_eval(*inputs)
    return results


# Salvataggio delle dimensioni delle matrici dinamiche per uso futuro
[Dyn_row, Dyn_col] = COEF_Dyn_loaded.shape
[RHS_row, RHS_col] = RHS_Dyn_loaded.shape

# Definizione delle variabili simboliche
all_symbolic_var = (u1, u2, u3, u4, u7, u10, u13, u18, u21, u24, u27, u28, u31, u34, u37, u42, u45, u48, u51, u52,
                    u53, u54, q1, q2, q3, q4, q7, q10, q13, q18, q21, q24, q27, q28, q31, q34, q37, q42, q45, q48, q51, q52, q53, q54)

# Evaluazione delle celle della matrice COEF
COEF_eval_results = [[None for _ in range(Dyn_col)] for _ in range(Dyn_row)]
PolyCOEF_eval_results = [
    [None for _ in range(Dyn_col)] for _ in range(Dyn_row)]
for i in range(Dyn_row):
    for j in range(Dyn_col):
        # Converting symbolic matrix to numerical function
        fun_to_eval = sy.lambdify(
            all_symbolic_var, COEF_Dyn_loaded[i, j], 'numpy')
        Polyfun_to_eval = sy.lambdify(
            all_symbolic_var, PolyCOEF_Dyn_loaded[i, j], 'numpy')

        COEF_eval_results[i][j] = fun_evaluation(fun_to_eval)
        PolyCOEF_eval_results[i][j] = fun_evaluation(Polyfun_to_eval)

        print(f"Evaluating COEF matrix, cell: {i, j}")

# Evaluazione delle celle della matrice RHS
RHS_eval_results = [[None for _ in range(RHS_col)] for _ in range(RHS_row)]
PolyRHS_eval_results = [[None for _ in range(RHS_col)] for _ in range(RHS_row)]
for i in range(RHS_row):
    for j in range(RHS_col):
        # Converting symbolic matrix to numerical function
        fun_to_eval = sy.lambdify(
            all_symbolic_var, RHS_Dyn_loaded[i, j], 'numpy')
        Polyfun_to_eval = sy.lambdify(
            all_symbolic_var, PolyRHS_Dyn_loaded[i, j], 'numpy')

        RHS_eval_results[i][j] = fun_evaluation(fun_to_eval)
        PolyRHS_eval_results[i][j] = fun_evaluation(Polyfun_to_eval)

        print(f"Evaluating RHS matrix, cell: {i, j}")


# ------------------------------------------- DIFFERENCE MATRICES ------------------------------------------------------ #
print("Computing difference matrices...")

COEF_diff_results = [[None for _ in range(Dyn_col)] for _ in range(Dyn_row)]
RHS_diff_results = [[None for _ in range(RHS_col)] for _ in range(RHS_row)]


# Calcolo degli errori assoluti normalizzati per il valore reale
for i in range(Dyn_row):
    for j in range(Dyn_col):
        diff = np.abs((COEF_eval_results[i][j] - PolyCOEF_eval_results[i][j])/COEF_eval_results[i][j])
        COEF_diff_results[i][j] = diff
        

for i in range(RHS_row):
    for j in range(RHS_col):
        diff = np.abs((RHS_eval_results[i][j] - PolyRHS_eval_results[i][j])/RHS_eval_results[i][j])
        RHS_diff_results[i][j] = diff

print("✅ Difference matrices computed.")


# -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #
print('Saving datas...')

# Funzione di conversione per la matrice diff


def convert_diff_matrix_to_serializable(matrix):
    return [[list(cell) for cell in row] for row in matrix]


# Convertiamo le matrici
COEF_diff_serializable = convert_diff_matrix_to_serializable(COEF_diff_results)
RHS_diff_serializable = convert_diff_matrix_to_serializable(RHS_diff_results)

# Convertiamo anche i risultati delle matrici COEF e RHS
COEF_results_serializable = convert_diff_matrix_to_serializable(
    COEF_eval_results)
RHS_results_serializable = convert_diff_matrix_to_serializable(
    RHS_eval_results)
PolyCOEF_results_serializable = convert_diff_matrix_to_serializable(
    PolyCOEF_eval_results)
PolyRHS_results_serializable = convert_diff_matrix_to_serializable(
    PolyRHS_eval_results)

# Convertiamo anche il vettore dei tempi
time_values_serializable = list(time_values)

# Creiamo un dizionario da salvare
diff_data = {
    'time_values': time_values_serializable,
    'COEF_diff': COEF_diff_serializable,
    'RHS_diff': RHS_diff_serializable,
    'COEF_results': COEF_results_serializable,          # Aggiunto per salvare i risultati della matrice COEF
    'RHS_results': RHS_results_serializable,            # Aggiunto per salvare i risultati della matrice RHS
    'PolyCOEF_results': PolyCOEF_results_serializable,  # Aggiunto per salvare i risultati della matrice COEF
    'PolyRHS_results': PolyRHS_results_serializable     # Aggiunto per salvare i risultati della matrice RHS
}


# Salviamo tutto in un file JSONSS
with open('difference_matrices_Satellite.json', 'w') as f:
    json.dump(diff_data, f)

print("📁 Difference matrices and time vector saved as 'difference_matrices.json'")


# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #
end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')
