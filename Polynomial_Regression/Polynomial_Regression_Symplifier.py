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

print('874')

# ---------------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------- #

start = time.perf_counter()

# ---------------------------------------------- INTERPOLATION OPTIONS ----------------------------------------------- #

# DataSet Options
t_span = 10                  # Defining the time span for the data set value generation in [s]
t_nval = 1000                # Defining the n° of values for each variable
t_step = t_span/t_nval       # Calculating the time steps of evaluation

# Interpolation options
r2_threshold = 0.999
max_degree = 4               # Il massimo grado per il polinomiale
threshold = 1e-4             # Threshold to set the polynomial coefficient equal to zero 



# ---------------------------------------- LOADING THE DYNAMICAL MATRICES -------------------------------------------- #
print('Loading dynamical matrices...')


# Reading the Json file
with open('Matrix_Kane/Clearspace_Satellite/Dynamical_Matrices_4arms_Preloadedsprings_dampers_FINAL.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']['COEF_Dyn']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']['RHS_Dyn']))



#------------------------------------------ SYMBOLIC VARIABLES ----------------------------------------------------#    
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
mshortarm = sy.symbols('mshortarm') # Robotic Short arm piece mass [kg]
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

c11 = sy.symbols('c11')             # B11 body damping coefficient [(N*m/s)/(rad/s)]
c21 = sy.symbols('c21')             # B21 body damping coefficient [(N*m/s)/(rad/s)]
c31 = sy.symbols('c31')             # B31 body damping coefficient [(N*m/s)/(rad/s)]
c41 = sy.symbols('c41')             # B41 body damping coefficient [(N*m/s)/(rad/s)]
c12 = sy.symbols('c12')             # B12 body damping coefficient [(N*m/s)/(rad/s)]
c22 = sy.symbols('c22')             # B22 body damping coefficient [(N*m/s)/(rad/s)]
c32 = sy.symbols('c32')             # B32 body damping coefficient [(N*m/s)/(rad/s)]
c42 = sy.symbols('c42')             # B42 body damping coefficient [(N*m/s)/(rad/s)]
c13 = sy.symbols('c13')             # B13 body damping coefficient [(N*m/s)/(rad/s)]
c23 = sy.symbols('c23')             # B23 body damping coefficient [(N*m/s)/(rad/s)]
c33 = sy.symbols('c33')             # B33 body damping coefficient [(N*m/s)/(rad/s)]
c43 = sy.symbols('c43')             # B43 body damping coefficient [(N*m/s)/(rad/s)]
c14 = sy.symbols('c14')             # B14 body damping coefficient [(N*m/s)/(rad/s)]
c24 = sy.symbols('c24')             # B24 body damping coefficient [(N*m/s)/(rad/s)]
c34 = sy.symbols('c34')             # B34 body damping coefficient [(N*m/s)/(rad/s)]
c44 = sy.symbols('c44')             # B44 body damping coefficient [(N*m/s)/(rad/s)]


#  ---------------------------------------------- #
# Initializing the gen.coordinates and speed
q1, q2, q3, q4, q7, q10, q13, q18, q21, q24, q27, q28, q31, q34, q37, q42, q45, q48, q51, q52, q53, q54 = sy.symbols('q1 q2 q3 q4 q7 q10 q13 q18 q21 q24 q27 q28 q31 q34 q37 q42 q45 q48 q51 q52 q53 q54')
u1, u2, u3, u4, u7, u10, u13, u18, u21, u24, u27, u28, u31, u34, u37, u42, u45, u48, u51, u52, u53, u54 = sy.symbols('u1 u2 u3 u4 u7 u10 u13 u18 u21 u24 u27 u28 u31 u34 u37 u42 u45 u48 u51 u52 u53 u54')

# Putting them into list to manage them better later
qi_list = [q1, q2, q3, q4, q7, q10, q13, q18, q21, q24, q27, q28, q31, q34, q37, q42, q45, q48, q51, q52, q53, q54]
ui_list = [u1, u2, u3, u4, u7, u10, u13, u18, u21, u24, u27, u28, u31, u34, u37, u42, u45, u48, u51, u52, u53, u54]


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
    T11 : 0,
    T21 : 0,
    T31 : 0,
    T41 : 0,
    T12 : 0,
    T22 : 0,
    T32 : 0,
    T42 : 0,
    T13 : 0,
    T23 : 0,
    T33 : 0,
    T43 : 0,
    T14 : 0,
    T24 : 0,
    T34 : 0,
    T44 : 0,

    
    # Defining the spring and damper coefficients
    k11 : 0.52,
    k21 : 0.3,
    k31 : 0.2,
    k41 : 0.1,
    k12 : 0.52,
    k22 : 0.3,
    k32 : 0.2,
    k42 : 0.1,
    k13 : 0.52,
    k23 : 0.3,
    k33 : 0.2,
    k43 : 0.1,
    k14 : 0.52,
    k24 : 0.3,
    k34 : 0.2,
    k44 : 0.1,


    c11 : 1.2,
    c21 : 0.8,
    c31 : 0.6,
    c41 : 0.4,
    c12 : 1.2,
    c22 : 0.8,
    c32 : 0.6,
    c42 : 0.4,
    c13 : 1.2,
    c23 : 0.8,
    c33 : 0.6,
    c43 : 0.4,
    c14 : 1.2,
    c24 : 0.8,
    c34 : 0.6,
    c44 : 0.4,


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
      #u1,  u2,  u3,  u4, u7,  u10, u13, u18, u21  u24  u27  u28  u31  u34  u37  u42  u45  u48  u51  u52  u53  u54  q1   q2   q3   q4   q7   q10  q13  q18  q21  q24  q27  q28  q31  q34  q37  q42  q45  q48  q51  q52  q53  q54
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
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
all_symbolic_var = (u1,u2,u3,u4,u7,u10,u13,u18,u21,u24,u27,u28,u31,u34,u37,u42,u45,u48,u51,u52,u53,u54,q1,q2,q3,q4,q7,q10,q13,q18,q21,q24,q27,q28,q31,q34,q37,q42,q45,q48,q51,q52,q53,q54)

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


# Funzione per trovare il miglior modello polinomiale
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
input_features = ['u1','u2','u3','u4','u7','u10','u13','u18','u21','u24','u27','u28',
                  'u31','u34','u37','u42','u45','u48','u51','u52','u53','u54',
                  'q1','q2','q3','q4','q7','q10','q13','q18','q21','q24','q27','q28',
                  'q31','q34','q37','q42','q45','q48','q51','q52','q53','q54']


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


# -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #
print('Saving datas...')

# Json Python necessities

# Converting the matrices in dictionaries
# Dynamical Matrices
COEF_Dyn_dict = {'COEF_Dyn_poly': str(COEF_poly_sympy)}
# RHS_Dyn_dict = {'RHS_Dyn_poly': str(RHS_poly_sympy)}

RHS_Dyn_dict = {'RHS_Dyn_poly': str(RHS_Dyn_loaded)}


# Writing and saving these dictionaries in a Json file
with open('Dynamical_Matrices_4arms_Preloadedsprings_dampers_PolyInterp_R0999.json', 'w') as json_file:
    json.dump({'COEF_Dyn': COEF_Dyn_dict, 'RHS_Dyn': RHS_Dyn_dict}, json_file)



# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')







