#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI-----------------------------------------------#
#------------------------------------------- SCRIPT OBJECTIVE ---------------------------------------------------------#

# The objective of this script is to solve the eq. of motion of a system by solving numerically the linear matrix system COEF = udot*RHS




#-------------------------------------------- IMPORTING THE NECESSARY PACKAGES ------------------------------------------#

import numpy as np
import sympy as sy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import json
import sympy.physics.mechanics as me
import json


#Cleaning the terminal 
os.system('cls')



#--------------------------------------------------- LOADING THE MATRICES ----------------------------------------------#

# Reading the Json file
with open('Dynamical_Matrices_SpinningSatellite_test.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)

with open('Kinematical_Matrices_SpinningSatellite_test.json', 'r') as json_file:
    data_Mat_Cin = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']['COEF_Dyn']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']['RHS_Dyn']))

# Reconstructing the Kinematical matrices
COEF_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['COEF_Kin']['COEF_Kin']))
RHS_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['RHS_Kin']['RHS_Kin']))



# --------------------------------------------------------------- #

# Initializing the variables
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
F = sy.symbols('F')
T = sy.symbols('T')

# Initializing the gen.coordinates and speed
q1, q2, q3, q52, q53, q54 = sy.symbols('q1 q2 q3 q52 q53 q54')
u1, u2, u3, u52, u53, u54 = sy.symbols('u1 u2 u3 u52 u53 u54')


# ----------------------------------------- SOLVING THE SYSTEM ---------------------------------------------------- #

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
    mlongarm: 0.836,        # Robotic Long arm piece mass [kg]
    mshortarm: 0.540,       # Robotic Short arm piece mass [kg]
    mbus: 122.8,            # Spacecraft bus mass [kg]
    F: 0,
    T: 0,


}


# Substituing the symbolic variables with their values inside the matrices
# Dynamical Matrices
COEF_Dyn_loaded = COEF_Dyn_loaded.xreplace(substitution)
RHS_Dyn_loaded = RHS_Dyn_loaded.xreplace(substitution)
COEF_Kin_loaded = COEF_Kin_loaded.xreplace(substitution)
RHS_Kin_loaded = RHS_Kin_loaded.xreplace(substitution)


#--------------------------------------------------- CALCULATIONS -------------------------------------------------------#

# We assemble the giant mass matrix for the system we want to solve, its composed by the mass matrix of the dynamical and 
# kinematicl equations. The same we do to assemble the giant forcing vector
Mass_Matrix = sy.BlockDiagMatrix(COEF_Dyn_loaded, COEF_Kin_loaded)
Forcing_Vector = sy.Matrix.vstack(RHS_Dyn_loaded, RHS_Kin_loaded)


# Transforming the matrices from "blockmatrices" to "normal matrices"
Mass_Matrix_dense = Mass_Matrix.as_mutable()
Forcing_Vector_dense = Forcing_Vector.as_mutable()

# Translating the symbolic matrices from sympy to numpy
all_symbolic_var = (u1,u2,u3,u52,u53,u54,q1,q2,q3,q52,q53,q54,T)
Mass_Matrix_lambdified = sy.lambdify(all_symbolic_var, Mass_Matrix_dense, 'numpy')
Forcing_Vector_lambdified = sy.lambdify(all_symbolic_var, Forcing_Vector_dense,'numpy')



def system(t, State_Vector):

    # ------------------------------ #
    # Unpacking 
    u1 = State_Vector[0]
    u2 = State_Vector[1]
    u3 = State_Vector[2]
    u52 = State_Vector[3]
    u53 = State_Vector[4]
    u54 = State_Vector[5]

    q1 = State_Vector[6]
    q2 = State_Vector[7]
    q3 = State_Vector[8]
    q52 = State_Vector[9]
    q53 = State_Vector[10]
    q54 = State_Vector[11]


#    # ------------------------------ #
   # Force and Torques timed application
    
    # # Torque applied for 1s<t<2.5s
    # if t > 1 and t < 3.0:
        
    #     T = 3

    # else:
    #     T = 0
    # ------------------------------ #
    # Matrix and Vector Evaluation
    Mass_Matrix_eval = Mass_Matrix_lambdified(u1,u2,u3,u52,u53,u54,q1,q2,q3,q52,q53,q54,T)
    Forcing_Vector_eval = Forcing_Vector_lambdified(u1,u2,u3,u52,u53,u54,q1,q2,q3,q52,q53,q54,T)

    # ------------------------------ #
    # Solving the system
    try:
        
        # Solve the linear system without explicitly calculating the inverse
        State_Vector_dot = np.linalg.solve(Mass_Matrix_eval, Forcing_Vector_eval)
        
    except np.linalg.LinAlgError:
        
        # If the matrix is singular, you can use the pseudo-inverse
        State_Vector_dot = np.dot(np.linalg.pinv(Mass_Matrix_eval), Forcing_Vector_eval)

    # ------------------------------ #

    print(t)

    return State_Vector_dot.flatten()


# Initial conditions
      #u1,  u2,  u3,  u52  u53  u54  q1   q2  q3   q52   q53  q54    
y0 = [0.3, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
      #0    1     2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26  27   28   29   30    31   32   33   34   35   36   37   38   39   40   41   42   43  

# Time interval for solution
t_span = (0.0, 20) 
t_eval = np.linspace(t_span[0], t_span[1], 2000) 

# Numerical resolution of the system with solve_ivp
solution = solve_ivp(system, t_span, y0, t_eval=t_eval, vectorized=False, method='RK23', rtol=1e-4, atol=1e-5, dense_output=True, max_step = 0.3)




# -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #
# Json Python necessities

data_to_save = {
    "t": solution.t.tolist(),  # Converting the numpy array in a list
    "y": solution.y.tolist()   # Converting the numpy array in a list
}

# Saving the datas in a json file
with open('solution_spinningspacecraft_test.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

print("Dati salvati con successo in solution.json")

# ---------------------------------------------------------------------------------------------------------------- #


print('Codice terminato')
