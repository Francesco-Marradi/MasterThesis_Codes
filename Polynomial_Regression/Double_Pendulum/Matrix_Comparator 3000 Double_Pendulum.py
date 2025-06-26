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
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor, parse_expr


import sys
sys.setrecursionlimit(100000000)

# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

#Cleaning the terminal 
os.system('cls')

print('543')

# ---------------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------- #

start = time.perf_counter()

# -------------------------------------------------------------------------------------------------------------------- #



# ---------------------------------------- LOADING THE DYNAMICAL MATRICES -------------------------------------------- #
print('Loading dynamical matrices...')


# Reading the Json file
with open('Matrix_Kane/Kane_MatrixForm_Double_Pendulum/Dynamical_Matrices_DoublePendulum.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)


# Reading the Json file
with open('Polynomial_Regression/Double_Pendulum/Results/Double_Pendulum_PolyInterp_R0999.json', 'r') as json_file:
    data_PolyMat_Dyn = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']))
PolyCOEF_Dyn_loaded = sy.Matrix(sy.sympify(data_PolyMat_Dyn['COEF_Dyn']))
PolyRHS_Dyn_loaded = sy.Matrix(sy.sympify(data_PolyMat_Dyn['RHS_Dyn']))



#------------------------------------------ SYMBOLIC VARIABLES ----------------------------------------------------#    
print('Defining symbols...')


# Initial motion parameters
t = sy.symbols('t')                # Definition of the time variable
g = sy.symbols('g')                # Gravitational acceleration [m/s^2]
m = sy.symbols('m')                # Particles Mass [kg]
l = sy.symbols('l')                # Rod length [m]
T = sy.symbols('T')                # Torque applied
F = sy.symbols('F')                # Force applied


#  ---------------------------------------------- #
# Initializing the gen.coordinates and speed
q6, q9 = sy.symbols('q6 q9')
u6, u9 = sy.symbols('u6 u9')

# Putting them into list to manage them better later
qi_list = [q6, q9]
ui_list = [u6, u9]



# ------------------------------------------------ SUBSTITUTING CONSTANTS -------------------------------------------------- #
print('Substituting constants...')

# Now we have to create a dictionary for the symbolic variables substituition inside the matrices. Practically we need
# to specify the masses, lenghts, ..., values for the system parameters

# Dictionary
substitution = {
    g: 9.81,     # Accelerazione gravitazionale [m/s^2]
    m : 1,       # Mass of the cart [kg]
    l : 2,       # Rod length [m]
    T : 0.0,
    F : 0.0,
}

# Substituing the symbolic variables with their values inside the matrices
# Dynamical Matrices
COEF_Dyn_loaded = COEF_Dyn_loaded.xreplace(substitution)
RHS_Dyn_loaded = RHS_Dyn_loaded.xreplace(substitution)



# ----------------------------------- GENERATING DATA SETS TO EVALUATE EACH CELL FUNCTION ---------------------------------- #
print('Generating data sets...')

# Now we need to load each cell's function, and apply to it a randomic value for the gen.coordinates and speeds, to 
# evaluate it, and so, create the data set for the polynomial regression.


# First, we need to specify the initial conditions for evaluating it
# Condizioni iniziali
      #u6,  u9,  q6,  q9 
y0 = [0.0, 0.0, 0.0, np.pi/4]
      #0    1    2   3     4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26  27   28   29   30    31   32   33   34   35   36   37   38   39   40   41   42   43  


# ------------------------------------------------------------------------------------------ #
# Constructing the gen.coordinates and speeds data values to evaluate each cell's function

t_span = 10                  # Defining the time span for the data set value generation in [s]
t_nval = 1000                # Defining the n° of values for each variable
t_step = t_span/t_nval      # Calculating the time steps of evaluation

time_values = np.linspace(0, t_span, t_nval) # Used for saving the Data_set


# Initializing the matrix containing the data set values
Data_set_mat = np.zeros((len(ui_list) + len(qi_list), t_nval))

# Defining the row half number
row_number = len(y0) // 2 


# Defining the for cycle to construct the data values to evaluate the cell's functions

for i in range(t_nval):                        # "i" moves along the matrix coloums

    for j in range(row_number):                # "j" moves along half of the matrix rows

        if i == 0:
            Data_set_mat[:row_number, i] = y0[:row_number]    # Assigning the initial condition values to the ui
            Data_set_mat[row_number:, i] = y0[row_number:]    # Assigning the initial condition values to the qi
            continue

        # ----------------------------------------------------- #

        # Defining the functions to randomic associate a value to the gen.coordinates
        # To not let these values make a very "spiky" dataset, they are going to be weighted by the time step

        # Calculating the gen.coordinate values, following the formula: qi(t+dt) = qi_random*dt + qi(t)
        Data_set_mat[j+row_number, i] = (np.random.uniform(-np.pi, np.pi))*t_step + Data_set_mat[j+row_number, i-1]

        # Calculating the gen.speed values, following the formula: ui(t+dt) = (qi(t+dt) - qi(t))/dt
        Data_set_mat[j, i] = (Data_set_mat[j+row_number, i] - Data_set_mat[j+row_number, i-1])/t_step
        
        # Defining an exception for the linear velocities of the root body
        if j in (row_number-3, row_number-2, row_number-1):

            # Calculating the gen.coordinate values, following the formula: qi(t+dt) = qi_random*dt + qi(t)
            Data_set_mat[j+row_number, i] = (np.random.uniform(-1, 1))*t_step + Data_set_mat[j+row_number, i-1]

            # Calculating the gen.speed values, following the formula: ui(t+dt) = (qi(t+dt) - qi(t))/dt
            Data_set_mat[j, i] = (Data_set_mat[j+row_number, i] - Data_set_mat[j+row_number, i-1])/t_step




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
all_symbolic_var = (u6,u9,q6,q9)

# Evaluazione delle celle della matrice COEF
COEF_eval_results = [[None for _ in range(Dyn_col)] for _ in range(Dyn_row)]
PolyCOEF_eval_results = [[None for _ in range(Dyn_col)] for _ in range(Dyn_row)]
for i in range(Dyn_row):
    for j in range(Dyn_col):
        # Converting symbolic matrix to numerical function
        fun_to_eval = sy.lambdify(all_symbolic_var, COEF_Dyn_loaded[i, j], 'numpy')
        Polyfun_to_eval = sy.lambdify(all_symbolic_var, PolyCOEF_Dyn_loaded[i, j], 'numpy')
        
        COEF_eval_results[i][j] = fun_evaluation(fun_to_eval)
        PolyCOEF_eval_results[i][j] = fun_evaluation(Polyfun_to_eval)

        print(f"Evaluating COEF matrix, cell: {i, j}")

# Evaluazione delle celle della matrice RHS
RHS_eval_results = [[None for _ in range(RHS_col)] for _ in range(RHS_row)]
PolyRHS_eval_results = [[None for _ in range(RHS_col)] for _ in range(RHS_row)]
for i in range(RHS_row):
    for j in range(RHS_col):
        # Converting symbolic matrix to numerical function
        fun_to_eval = sy.lambdify(all_symbolic_var, RHS_Dyn_loaded[i, j], 'numpy')
        Polyfun_to_eval = sy.lambdify(all_symbolic_var, PolyRHS_Dyn_loaded[i, j], 'numpy')

        RHS_eval_results[i][j] = fun_evaluation(fun_to_eval)
        PolyRHS_eval_results[i][j] = fun_evaluation(Polyfun_to_eval)

        print(f"Evaluating RHS matrix, cell: {i, j}")



# ------------------------------------------- DIFFERENCE MATRICES ------------------------------------------------------ #
print("Computing difference matrices...")

# Inizializza le matrici
COEF_diff_results = [[None for _ in range(Dyn_col)] for _ in range(Dyn_row)]
RHS_diff_results = [[None for _ in range(RHS_col)] for _ in range(RHS_row)]

# # Inizializza le matrici COEF e RHS che conterranno gli errori massimi
# COEF_max_errors = np.zeros((Dyn_row, Dyn_col))
# RHS_max_errors = np.zeros((RHS_row, RHS_col))

# Calcolo degli errori assoluti normalizzati per il valore reale
for i in range(Dyn_row):
    for j in range(Dyn_col):
        diff = np.abs((COEF_eval_results[i][j] - PolyCOEF_eval_results[i][j])/COEF_eval_results[i][j])
        COEF_diff_results[i][j] = diff
        # COEF_max_errors[i, j] = np.max(diff)

for i in range(RHS_row):
    for j in range(RHS_col):
        diff = np.abs((RHS_eval_results[i][j] - PolyRHS_eval_results[i][j])/RHS_eval_results[i][j])
        RHS_diff_results[i][j] = diff
        # RHS_max_errors[i, j] = np.max(diff)

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
COEF_results_serializable = convert_diff_matrix_to_serializable(COEF_eval_results)
RHS_results_serializable = convert_diff_matrix_to_serializable(RHS_eval_results)
PolyCOEF_results_serializable = convert_diff_matrix_to_serializable(PolyCOEF_eval_results)
PolyRHS_results_serializable = convert_diff_matrix_to_serializable(PolyRHS_eval_results)

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
with open('difference_matrices_doublependulum.json', 'w') as f:
    json.dump(diff_data, f)

print("📁 Difference matrices and time vector saved as 'difference_matrices.json'")



# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #
end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')







