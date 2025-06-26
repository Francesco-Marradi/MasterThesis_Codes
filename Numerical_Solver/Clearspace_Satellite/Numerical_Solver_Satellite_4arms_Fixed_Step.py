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
import time

#Cleaning the terminal 
os.system('cls')

print('ciao123')

# ---------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

start = time.perf_counter()



#--------------------------------------------------- LOADING THE MATRICES ----------------------------------------------#
print('Loading dynamical matrices...')


# Reading the Json file
with open('Dynamical_Matrices_4arms_Preloadedsprings_dampers_PolyInterp.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)

with open('Kinematical_Matrices_4arms_Preloadedsprings_dampers_FINAL.json', 'r') as json_file:
    data_Mat_Cin = json.load(json_file)



print('Reconstructing dynamical matrices...')
# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']['COEF_Dyn_poly']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']['RHS_Dyn_poly']))

# Reconstructing the Kinematical matrices
COEF_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['COEF_Kin']['COEF_Kin']))
RHS_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['RHS_Kin']['RHS_Kin']))



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



#--------------------------------------------------- CALCULATIONS -------------------------------------------------------#

def rk3_fixed_step(f, y0, t_span, h):
    """
    Runge-Kutta 3rd order with fixed step size.
    
    Parameters:
    - f: function(t, y) -> dy/dt (dimension: len(y))
    - y0: initial state (1D array)
    - t_span: tuple (t0, tf)
    - h: step size
    
    Returns:
    - t: array of time points
    - y: array of state vectors at each time step (shape: [N_steps, len(y0)])
    """
    t0, tf = t_span
    t_values = np.arange(t0, tf + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    
    y = np.array(y0, dtype=float)

    for i, t in enumerate(t_values):
        y_values[i] = y
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h, y - h * k1 + 2 * h * k2)
        y = y + h / 6 * (k1 + 4 * k2 + k3)

    return t_values, y_values


print('Preparing the matrices for integration...')

# We assemble the giant mass matrix for the system we want to solve, its composed by the mass matrix of the dynamical and 
# kinematicl equations. The same we do to assemble the giant forcing vector
Mass_Matrix = sy.BlockDiagMatrix(COEF_Dyn_loaded, COEF_Kin_loaded)
Forcing_Vector = sy.Matrix.vstack(RHS_Dyn_loaded, RHS_Kin_loaded)


# Transforming the matrices from "blockmatrices" to "normal matrices"
Mass_Matrix_dense = Mass_Matrix.as_mutable()
Forcing_Vector_dense = Forcing_Vector.as_mutable()

# Translating the symbolic matrices from sympy to numpy
all_symbolic_var = (u1,u2,u3,u4,u7,u10,u13,u18,u21,u24,u27,u28,u31,u34,u37,u42,u45,u48,u51,u52,u53,u54,q1,q2,q3,q4,q7,q10,q13,q18,q21,q24,q27,q28,q31,q34,q37,q42,q45,q48,q51,q52,q53,q54)
Mass_Matrix_lambdified = sy.lambdify(all_symbolic_var, Mass_Matrix_dense, 'numpy')
Forcing_Vector_lambdified = sy.lambdify(all_symbolic_var, Forcing_Vector_dense,'numpy')


print('Starting the numerical integrator...')

def system(t, State_Vector):

    # ------------------------------ #
    # Unpacking 
    u1 = State_Vector[0]
    u2 = State_Vector[1]
    u3 = State_Vector[2]
    u4 = State_Vector[3]
    u7 = State_Vector[4]
    u10 = State_Vector[5]
    u13 = State_Vector[6]
    u18 = State_Vector[7]
    u21 = State_Vector[8]
    u24 = State_Vector[9]
    u27 = State_Vector[10]
    u28 = State_Vector[11]
    u31 = State_Vector[12]
    u34 = State_Vector[13]
    u37 = State_Vector[14]
    u42 = State_Vector[15]
    u45 = State_Vector[16]
    u48 = State_Vector[17]
    u51 = State_Vector[18]
    u52 = State_Vector[19]
    u53 = State_Vector[20]
    u54 = State_Vector[21]


    q1 = State_Vector[22]
    q2 = State_Vector[23]
    q3 = State_Vector[24]
    q4 = State_Vector[25]
    q7 = State_Vector[26]
    q10 = State_Vector[27]
    q13 = State_Vector[28]
    q18 = State_Vector[29]
    q21 = State_Vector[30]
    q24 = State_Vector[31]
    q27 = State_Vector[32]
    q28 = State_Vector[33]
    q31 = State_Vector[34]
    q34 = State_Vector[35]
    q37 = State_Vector[36]
    q42 = State_Vector[37]
    q45 = State_Vector[38]
    q48 = State_Vector[39]
    q51 = State_Vector[40]
    q52 = State_Vector[41]
    q53 = State_Vector[42]
    q54 = State_Vector[43]


    # ------------------------------ #
    # Matrix and Vector Evaluation
    Mass_Matrix_eval = Mass_Matrix_lambdified(u1,u2,u3,u4,u7,u10,u13,u18,u21,u24,u27,u28,u31,u34,u37,u42,u45,u48,u51,u52,u53,u54,q1,q2,q3,q4,q7,q10,q13,q18,q21,q24,q27,q28,q31,q34,q37,q42,q45,q48,q51,q52,q53,q54)
    Forcing_Vector_eval = Forcing_Vector_lambdified(u1,u2,u3,u4,u7,u10,u13,u18,u21,u24,u27,u28,u31,u34,u37,u42,u45,u48,u51,u52,u53,u54,q1,q2,q3,q4,q7,q10,q13,q18,q21,q24,q27,q28,q31,q34,q37,q42,q45,q48,q51,q52,q53,q54)


    # ------------------------------ #
    # Solving the system

    cond = np.linalg.cond(Mass_Matrix_eval)
    if cond > 1e10:
        print(f"Condizionamento elevato aiutoo: {cond:.2e}")
        input()

    try:
        
        # Solve the linear system without explicitly calculating the inverse
        State_Vector_dot = np.linalg.solve(Mass_Matrix_eval, Forcing_Vector_eval)
        
    except np.linalg.LinAlgError:
        
        # If the matrix is singular, you can use the pseudo-inverse
        State_Vector_dot = np.dot(np.linalg.pinv(Mass_Matrix_eval), Forcing_Vector_eval)


    # ------------------------------ #
    # Return the derivatives of all variables
    print(t)

    return State_Vector_dot.flatten()


# Initial conditions
      #u1,  u2,  u3,  u4, u7,  u10, u13, u18, u21  u24  u27  u28  u31  u34  u37  u42  u45  u48  u51  u52  u53  u54  q1   q2   q3   q4   q7   q10  q13  q18  q21  q24  q27  q28  q31  q34  q37  q42  q45  q48  q51  q52  q53  q54
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
      #0    1    2   3     4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26  27   28   29   30    31   32   33   34   35   36   37   38   39   40   41   42   43  

# Time interval for solution
t_span = (0.0, 8.5) 
h = 0.001  # fixed integration time step

# Numerical resolution of the system with solve_ivp
t_vals, y_vals = rk3_fixed_step(system, y0, t_span, h)



# -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #
print('Saving datas...')


# Json Python necessities

data_to_save = {
    "t": t_vals.tolist(),  # Converting the numpy array in a list
    "y": y_vals.tolist()   # Converting the numpy array in a list
}

# Saving the datas in a json file
with open('solution_4arms_Preloadedsprings_dampers_Fixed_step.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

print("Dati salvati con successo in solution.json")

# ---------------------------------------------------------------------------------------------------------------- #



# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')