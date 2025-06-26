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
with open('Dynamical_Matrices_1arm_fixedbus_Preloadedsprings_dampers.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)

with open('Kinematical_Matrices_1arm_fixedbus_Preloadedsprings_dampers.json', 'r') as json_file:
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


# Forces and Torques applied
# Robotic Arm 1
T11 = sy.symbols('T11')             # Torque acting on the arms [N*m]
T21 = sy.symbols('T21')             # Torque acting on the arms [N*m]
T31 = sy.symbols('T31')             # Torque acting on the arms [N*m]
T41 = sy.symbols('T41')             # Torque acting on the arms [N*m]
F = sy.symbols('F')                 # Force acting on the arms [N]


# Initializing the gen.coordinates and speed
q4, q7, q10, q13 = sy.symbols('q4 q7 q10 q13')
u4, u7, u10, u13 = sy.symbols('u4 u7 u10 u13')


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
    F: 0.0,                 # Force acting on the arms [N]
    mlongarm: 0.836,        # Robotic Long arm piece mass [kg]
    mshortarm: 0.540,       # Robotic Short arm piece mass [kg]
    mbus: 122.8,            # Spacecraft bus mass [kg]
}


# Substituing the symbolic variables with their values inside the matrices
# Dynamical Matrices
COEF_Dyn_loaded = COEF_Dyn_loaded.xreplace(substitution)
RHS_Dyn_loaded = RHS_Dyn_loaded.xreplace(substitution)



#--------------------------------------------------- CALCULATIONS -------------------------------------------------------#

# We assemble the giant mass matrix for the system we want to solve, its composed by the mass matrix of the dynamical and 
# kinematicl equations. The same we do to assemble the giant forcing vector
Mass_Matrix = sy.BlockDiagMatrix(COEF_Dyn_loaded, COEF_Kin_loaded)
Forcing_Vector = sy.Matrix.vstack(RHS_Dyn_loaded, RHS_Kin_loaded)


# Transforming the matrices from "blockmatrices" to "normal matrices"
Mass_Matrix_dense = Mass_Matrix.as_mutable()
Forcing_Vector_dense = Forcing_Vector.as_mutable()

# Translating the symbolic matrices from sympy to numpy
all_symbolic_var = (u4,u7,u10,u13,q4,q7,q10,q13,T11,T21,T31,T41)
Mass_Matrix_lambdified = sy.lambdify(all_symbolic_var, Mass_Matrix_dense, 'numpy')
Forcing_Vector_lambdified = sy.lambdify(all_symbolic_var, Forcing_Vector_dense,'numpy')



def system(t, State_Vector):

    # ------------------------------ #
    # Unpacking 
    u4 = State_Vector[0]
    u7 = State_Vector[1]
    u10 = State_Vector[2]
    u13 = State_Vector[3]

    q4 = State_Vector[4]
    q7 = State_Vector[5]
    q10 = State_Vector[6]
    q13 = State_Vector[7]


    # ------------------------------ #
    # Force and Torques timed application
    
    # Torque applied for 1s<t<2.5s
    if t > 1 and t < 2:
        # Robotic arm 1
        T11 = 0.05
        T21 = 0.01
        T31 = 0.025
        T41 = 0.005

    else:
        # Robotic arm 1
        T11 = 0.0
        T21 = 0.0
        T31 = 0.0
        T41 = 0.0

    # ------------------------------ #
    # Matrix and Vector Evaluation
    Mass_Matrix_eval = Mass_Matrix_lambdified(u4,u7,u10,u13,q4,q7,q10,q13,T11,T21,T31,T41)
    Forcing_Vector_eval = Forcing_Vector_lambdified(u4,u7,u10,u13,q4,q7,q10,q13,T11,T21,T31,T41)

    # ------------------------------ #
    # Solving the system
    try:
        
        # Solve the linear system without explicitly calculating the inverse
        State_Vector_dot = np.linalg.solve(Mass_Matrix_eval, Forcing_Vector_eval)
        
    except np.linalg.LinAlgError:
        
        # If the matrix is singular, you can use the pseudo-inverse
        State_Vector_dot = np.dot(np.linalg.pinv(Mass_Matrix_eval), Forcing_Vector_eval)

    # ------------------------------ #
    # For the saturation thing to work, i need also to set to 0 the acceleration of each body, 
    # when it achieves the saturation condition.

    # Robotic Arm 1
    if q4 < (-45)*(np.pi/180):    # q4 angle saturation condition
        State_Vector_dot[0] = 0   # blocking the acceleration of B11 
        State_Vector_dot[4] = 0   # blocking the velocity of B11 
        
    if q7 < (-105)*(np.pi/180):   # q7 angle saturation condition
        State_Vector_dot[1] = 0   # blocking the acceleration of B21 
        State_Vector_dot[5] = 0   # blocking the angle of B21

    if q10 < (-75)*(np.pi/180):   # q10 angle saturation condition
        State_Vector_dot[2] = 0   # blocking the acceleration of B31 
        State_Vector_dot[6] = 0   # blocking the angle of B31

    if q13 < (-15)*(np.pi/180):   # q13 angle saturation condition
        State_Vector_dot[3] = 0   # blocking the acceleration of B41 
        State_Vector_dot[7] = 0   # blocking the angle of B41


    # ------------------------------ #
    # Return the derivatives of all variables
    print(t)

    return State_Vector_dot.flatten()


# Initial conditions
    #  u4  u7   u10  u13   q4  q7   q10  q13  
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  


# Time interval for solution
t_span = (0.0, 10.0) 
t_eval = np.linspace(t_span[0], t_span[1], 500) 

# Numerical resolution of the system with solve_ivp
solution = solve_ivp(system, t_span, y0, t_eval=t_eval, vectorized=False, method='RK23', rtol=1e-4, atol=1e-5, dense_output=True, max_step=0.3)



# -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #
# Json Python necessities

data_to_save = {
    "t": solution.t.tolist(),  # Converting the numpy array in a list
    "y": solution.y.tolist()   # Converting the numpy array in a list
}

# Saving the datas in a json file
with open('solution_1arm_fixedbus_torqueapplied_dg.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

print("Dati salvati con successo in solution.json")

# ---------------------------------------------------------------------------------------------------------------- #


print('Codice terminato')
