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
import csv
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor, parse_expr


# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

#Cleaning the terminal
os.system('cls')

print('281')

# ---------------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------- #

start = time.perf_counter()

# ---------------------------------------------- INTERPOLATION OPTIONS ----------------------------------------------- #

# DataSet Options
t_span = 10                  # Defining the time span for the data set value generation in [s]
t_nval = 1000                # Defining the n° of values for each variable
t_step = t_span/t_nval       # Calculating the time steps of evaluation

# Interpolation options
r2_threshold = 0.999
max_degree = 5               # Il massimo grado per il polinomiale
threshold = 1e-4             # Threshold to set the polynomial coefficient equal to zero 


# ---------------------------------------- LOADING THE DYNAMICAL MATRICES -------------------------------------------- #
print('Loading dynamical matrices...')


# Reading the Json file
with open('Matrix_Kane/Kane_MatrixForm_Double_Pendulum/Dynamical_Matrices_DoublePendulum.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']))


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

# Now we are loading each cell function from the Dynamical matrices and evaluate it with the data set we have

def fun_evaluation(fun_to_eval):
    results = np.zeros(t_nval)

    for i in range(t_nval):
        # Estrazione dei dati dal dataset per ogni variabile
        inputs = [Data_set_mat[k, i] for k in range(Data_set_mat.shape[0])]

        # Valutazione della funzione e salvataggio del risultato
        results[i] = fun_to_eval(*inputs)

    return results



# Saving the Dynamical matrices dimensions, to use them later
[Dyn_row,Dyn_col] = COEF_Dyn_loaded.shape
[RHS_row,RHS_col] = RHS_Dyn_loaded.shape

# Variables 
all_symbolic_var = (u6,u9,q6,q9)

# Evaluating the COEF cells functions
COEF_eval_results = [[None for _ in range(Dyn_col)] for _ in range(Dyn_row)]

for i in range(Dyn_row):

    for j in range(Dyn_col):
        
        # Converting the Dynamical Matrices from sympy to numpy
        fun_to_eval = sy.lambdify(all_symbolic_var, COEF_Dyn_loaded[i, j], 'numpy')
        COEF_eval_results[i][j] = fun_evaluation(fun_to_eval)

        print(f"Evaluating COEF matrix, cell: {i,j}")

# Evaluating the RHS cells functions
RHS_eval_results = [[None for _ in range(RHS_col)] for _ in range(RHS_row)]

for i in range(RHS_row):

    for j in range(RHS_col):
        
        # Converting the Dynamical Matrices from sympy to numpy
        fun_to_eval = sy.lambdify(all_symbolic_var, RHS_Dyn_loaded[i, j], 'numpy')
        RHS_eval_results[i][j] = fun_evaluation(fun_to_eval)

        print(f"Evaluating RHS matrix, cell: {i,j}")



# ----------------------------------- POLYNOMIAL REGRESSION FOR EACH CELL FUNCTION RESULTS ------------------------------------------ #
print('Using adaptive polynomial regression to approximate the cells functions...')

COEF_poly_models = [[None for _ in range(Dyn_col)] for _ in range(Dyn_row)]
RHS_poly_models = [[None for _ in range(RHS_col)] for _ in range(RHS_row)]

COEF_feature_names = [[None for _ in range(Dyn_col)] for _ in range(Dyn_row)]
RHS_feature_names = [[None for _ in range(RHS_col)] for _ in range(RHS_row)]

# Nuovi flag
COEF_poly_valid = [[False for _ in range(Dyn_col)] for _ in range(Dyn_row)]
RHS_poly_valid = [[False for _ in range(RHS_col)] for _ in range(RHS_row)]


# Function to find the best polynomial
def find_best_polynomial_model(X, y, r2_threshold, max_degree, input_feature_names):
    for degree in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        r2 = model.score(X_poly, y)

        if r2 >= r2_threshold:
            feature_names = poly.get_feature_names_out(input_feature_names)
            return model, r2, degree, feature_names

    return None, 0.0, None, []

X = Data_set_mat.T
input_features = ['u6','u9','q9','q9']


# ------------------ COEF matrix regression ------------------
for i in range(Dyn_row):
    for j in range(Dyn_col):
        y = COEF_eval_results[i][j]
        model, r2, model_type, feature_names = find_best_polynomial_model(X, y, r2_threshold, max_degree, input_features)

        if model is None or r2 < r2_threshold:
            COEF_poly_models[i][j] = COEF_Dyn_loaded[i, j]
            COEF_poly_valid[i][j] = False
            print(f"⚠️ COEF[{i},{j}] - R² below threshold, fallback to original symbolic expression.")
        else:
            COEF_poly_models[i][j] = model
            COEF_feature_names[i][j] = feature_names
            COEF_poly_valid[i][j] = True
            print(f"COEF[{i},{j}] - Polynomial degree {model_type}, R² = {r2:.4f}")


# ------------------ RHS matrix regression ------------------
for i in range(RHS_row):
    for j in range(RHS_col):
        y = RHS_eval_results[i][j]
        model, r2, model_type, feature_names = find_best_polynomial_model(X, y, r2_threshold, max_degree, input_features)

        if model is None or r2 < r2_threshold:
            RHS_poly_models[i][j] = RHS_Dyn_loaded[i, j]
            RHS_poly_valid[i][j] = False
            print(f"⚠️ RHS[{i},{j}] - R² below threshold, fallback to original symbolic expression.")
        else:
            RHS_poly_models[i][j] = model
            RHS_feature_names[i][j] = feature_names
            RHS_poly_valid[i][j] = True
            print(f"RHS[{i},{j}] - Polynomial degree {model_type}, R² = {r2:.4f}")



# ------------------ RE-CONSTRUCTING SYMBOLIC POLYNOMIALS ------------------
print('Re-constructing the polynomials...')

variables = sy.symbols(input_features)
symbol_dict = dict(zip(input_features, variables))

def build_term_directly(name):
    parts = name.replace('^', '**').split(' ')
    return reduce(mul, [eval(part, {}, symbol_dict) for part in parts])

term_cache = {}
COEF_poly_sympy = sy.MutableDenseMatrix(Dyn_row, Dyn_col, [0]*Dyn_row*Dyn_col)
RHS_poly_sympy = sy.MutableDenseMatrix(RHS_row, RHS_col, [0]*RHS_row*RHS_col)


# -------------------------- COEF --------------------------
for i in range(Dyn_row):
    for j in range(Dyn_col):
        if not COEF_poly_valid[i][j]:
            COEF_poly_sympy[i, j] = COEF_Dyn_loaded[i, j]
            print(f"⚠️ COEF[{i},{j}] - Using original symbolic expression.")
            continue

        model = COEF_poly_models[i][j]
        feature_names = COEF_feature_names[i][j]
        coefs = model.coef_
        intercept = float(f"{model.intercept_:.6g}")  # Arrotonda anche l'intercetta
        coefs[np.abs(coefs) < threshold] = 0
        expr = intercept

        print(f"Re-constructing poly of COEF {i,j} cell")

        try:
            for name, coef in zip(feature_names, coefs):
                if coef == 0:
                    continue
                if name not in term_cache:
                    term_cache[name] = build_term_directly(name)
                coef_rounded = float(f"{coef:.6g}")  # Arrotonda a 4 cifre significative
                expr += coef_rounded * term_cache[name]
            COEF_poly_sympy[i, j] = expr
        except Exception as e:
            print(f"⚠️ COEF[{i},{j}] - Error building expression, using original: {e}")
            COEF_poly_sympy[i, j] = COEF_Dyn_loaded[i, j]

# -------------------------- RHS --------------------------
for i in range(RHS_row):
    for j in range(RHS_col):
        if not RHS_poly_valid[i][j]:
            RHS_poly_sympy[i, j] = RHS_Dyn_loaded[i, j]
            print(f"⚠️ RHS[{i},{j}] - Using original symbolic expression.")
            continue

        model = RHS_poly_models[i][j]
        feature_names = RHS_feature_names[i][j]
        coefs = model.coef_
        intercept = float(f"{model.intercept_:.4g}")
        coefs[np.abs(coefs) < threshold] = 0
        expr = intercept

        print(f"Re-constructing poly of RHS {i,j} cell")

        try:
            for name, coef in zip(feature_names, coefs):
                if coef == 0:
                    continue
                if name not in term_cache:
                    term_cache[name] = build_term_directly(name)
                coef_rounded = float(f"{coef:.4g}")
                expr += coef_rounded * term_cache[name]
            RHS_poly_sympy[i, j] = expr
        except Exception as e:
            print(f"⚠️ RHS[{i},{j}] - Error building expression, using original: {e}")
            RHS_poly_sympy[i, j] = RHS_Dyn_loaded[i, j]



end = time.perf_counter()
# -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #
print('Saving datas...')

# Json Python necessities

# Converting the matrices in dictionaries
# Dynamical Matrices
# COEF_Dyn_dict = {'COEF_Dyn_poly': str(COEF_poly_sympy)}
# RHS_Dyn_dict = {'RHS_Dyn_poly': str(RHS_poly_sympy)}

# RHS_Dyn_dict = {'RHS_Dyn_poly': str(RHS_Dyn_loaded)}


# Writing and saving these dictionaries in a Json file
with open('Double_Pendulum_PolyInterp_R0999.json', 'w') as json_file:
    json.dump({'COEF_Dyn': str(COEF_poly_sympy), 'RHS_Dyn': str(RHS_poly_sympy)}, json_file)



# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

# end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')







