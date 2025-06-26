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
with open('Dynamical_Matrices_4arms_torqueapplied.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)

with open('Kinematical_Matrices_4arms_torqueapplied.json', 'r') as json_file:
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
F = sy.symbols('F')                 # Force acting on the arms [N]


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
all_symbolic_var = (u1,u2,u3,u4,u7,u10,u13,u18,u21,u24,u27,u28,u31,u34,u37,u42,u45,u48,u51,u52,u53,u54,q1,q2,q3,q4,q7,q10,q13,q18,q21,q24,q27,q28,q31,q34,q37,q42,q45,q48,q51,q52,q53,q54,T11,T21,T31,T41,T12,T22,T32,T42,T13,T23,T33,T43,T14,T24,T34,T44)
Mass_Matrix_lambdified = sy.lambdify(all_symbolic_var, Mass_Matrix_dense, 'numpy')
Forcing_Vector_lambdified = sy.lambdify(all_symbolic_var, Forcing_Vector_dense,'numpy')



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
    # Force and Torques timed application
   
    # Torque applied for 1s<t<1.5s
    if t > 1 and t < 1.5:
        # Robotic arm 1
        T11 = 0.05
        T21 = 0.01
        T31 = 0.025
        T41 = 0.005

        # Robotic arm 2
        T12 = 0.05
        T22 = 0.01
        T32 = 0.025
        T42 = 0.005

        # Robotic arm 3
        T13 = 0.05
        T23 = 0.01
        T33 = 0.025
        T43 = 0.005

        # Robotic arm 4
        T14 = 0.05
        T24 = 0.01
        T34 = 0.025
        T44 = 0.005

    else:
        # Robotic arm 1
        T11 = 0.0
        T21 = 0.0
        T31 = 0.0
        T41 = 0.0

        # Robotic arm 2
        T12 = 0.0
        T22 = 0.0
        T32 = 0.0
        T42 = 0.0

        # Robotic arm 3
        T13 = 0.0
        T23 = 0.0
        T33 = 0.0
        T43 = 0.0

        # Robotic arm 4
        T14 = 0.0
        T24 = 0.0
        T34 = 0.0
        T44 = 0.0

    # ------------------------------ #
    # Matrix and Vector Evaluation
    Mass_Matrix_eval = Mass_Matrix_lambdified(u1,u2,u3,u4,u7,u10,u13,u18,u21,u24,u27,u28,u31,u34,u37,u42,u45,u48,u51,u52,u53,u54,q1,q2,q3,q4,q7,q10,q13,q18,q21,q24,q27,q28,q31,q34,q37,q42,q45,q48,q51,q52,q53,q54,T11,T21,T31,T41,T12,T22,T32,T42,T13,T23,T33,T43,T14,T24,T34,T44)
    Forcing_Vector_eval = Forcing_Vector_lambdified(u1,u2,u3,u4,u7,u10,u13,u18,u21,u24,u27,u28,u31,u34,u37,u42,u45,u48,u51,u52,u53,u54,q1,q2,q3,q4,q7,q10,q13,q18,q21,q24,q27,q28,q31,q34,q37,q42,q45,q48,q51,q52,q53,q54,T11,T21,T31,T41,T12,T22,T32,T42,T13,T23,T33,T43,T14,T24,T34,T44)

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
        State_Vector_dot[3] = 0   # blocking the acceleration of B11 
        State_Vector_dot[25] = 0  # blocking the velocity of B11 
        
    if q7 < (-105)*(np.pi/180):   # q7 angle saturation condition
        State_Vector_dot[4] = 0   # blocking the acceleration of B21 
        State_Vector_dot[26] = 0  # blocking the angle of B21

    if q10 < (-75)*(np.pi/180):   # q10 angle saturation condition
        State_Vector_dot[5] = 0   # blocking the acceleration of B31 
        State_Vector_dot[27] = 0  # blocking the angle of B31

    if q13 < (-15)*(np.pi/180):   # q13 angle saturation condition
        State_Vector_dot[6] = 0   # blocking the acceleration of B41 
        State_Vector_dot[28] = 0  # blocking the angle of B41


    # Robotic Arm 2
    if q18 > (45)*(np.pi/180):    # q18 angle saturation condition
        State_Vector_dot[7] = 0   # blocking the acceleration of B12 
        State_Vector_dot[29] = 0  # blocking the velocity of B12 
        
    if q21 > (105)*(np.pi/180):   # q21 angle saturation condition
        State_Vector_dot[8] = 0   # blocking the acceleration of B22 
        State_Vector_dot[30] = 0  # blocking the angle of B22

    if q24 > (75)*(np.pi/180):     # q24 angle saturation condition
        State_Vector_dot[9] = 0    # blocking the acceleration of B32 
        State_Vector_dot[31] = 0   # blocking the angle of B32

    if q27 > (15)*(np.pi/180):     # q27 angle saturation condition
        State_Vector_dot[10] = 0    # blocking the acceleration of B42
        State_Vector_dot[32] = 0   # blocking the angle of B42


    # Robotic Arm 3
    if q28 > (45)*(np.pi/180):     # q28 angle saturation condition
        State_Vector_dot[11] = 0   # blocking the acceleration of B13 
        State_Vector_dot[33] = 0   # blocking the velocity of B13
        
    if q31 > (105)*(np.pi/180):    # q31 angle saturation condition
        State_Vector_dot[12] = 0   # blocking the acceleration of B23 
        State_Vector_dot[34] = 0   # blocking the angle of B23

    if q34 > (75)*(np.pi/180):     # q34 angle saturation condition
        State_Vector_dot[13] = 0   # blocking the acceleration of B33 
        State_Vector_dot[35] = 0   # blocking the angle of B33

    if q37 > (15)*(np.pi/180):     # q37 angle saturation condition
        State_Vector_dot[14] = 0   # blocking the acceleration of B43 
        State_Vector_dot[36] = 0   # blocking the angle of B43


    # Robotic Arm 4
    if q42 < (-45)*(np.pi/180):    # q42 angle saturation condition
        State_Vector_dot[15] = 0   # blocking the acceleration of B11 
        State_Vector_dot[37] = 0   # blocking the velocity of B11 
        
    if q45 < (-105)*(np.pi/180):   # q45 angle saturation condition
        State_Vector_dot[16] = 0   # blocking the acceleration of B21 
        State_Vector_dot[38] = 0   # blocking the angle of B21SSSS

    if q48 < (-75)*(np.pi/180):    # q48 angle saturation condition
        State_Vector_dot[17] = 0   # blocking the acceleration of B31 
        State_Vector_dot[39] = 0   # blocking the angle of B31

    if q51 < (-15)*(np.pi/180):    # q51 angle saturation condition
        State_Vector_dot[18] = 0   # blocking the acceleration of B41 
        State_Vector_dot[40] = 0   # blocking the angle of B41



    # ------------------------------ #
    # Return the derivatives of all variables
    print(t)

    return State_Vector_dot.flatten()


# Initial conditions
      #u1,  u2,  u3,  u4, u7,  u10, u13, u18, u21  u24  u27  u28  u31  u34  u37  u42  u45  u48  u51  u52  u53  u54  q1   q2   q3   q4   q7   q10  q13  q18  q21  q24  q27  q28  q31  q34  q37  q42  q45  q48  q51  q52  q53  q54
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
      #0    1    2   3     4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26  27   28   29   30    31   32   33   34   35   36   37   38   39   40   41   42   43  

# Time interval for solution
t_span = (0.0, 7.0) 
t_eval = np.linspace(t_span[0], t_span[1], 500) 

# Numerical resolution of the system with solve_ivp
solution = solve_ivp(system, t_span, y0, t_eval=t_eval, vectorized=False, method='RK23', rtol=1e-4, atol=1e-5, dense_output=True, max_step = 0.3)




# -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #
# Json Python necessities

data_to_save = {
    "t": solution.t.tolist(),  # Converting the numpy array in a list
    "y": solution.y.tolist()   # Converting the numpy array in a list
}

# Saving the datas in a json file
with open('solution_4arms_torqueapplied.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

print("Dati salvati con successo in solution.json")

# ---------------------------------------------------------------------------------------------------------------- #


print('Codice terminato')
