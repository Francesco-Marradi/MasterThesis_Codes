#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI-----------------------------------------------#
#------------------------------------------- SCRIPT OBJECTIVE ---------------------------------------------------------#

# The objective of this script is to solve the eq. of motion of a system by solving numerically the linear matrix system COEF = udot*RHS




#-------------------------------------------- IMPORTING THE NECESSARY PACKAGES ------------------------------------------#

import numpy as np
import sympy as sy
from scipy.integrate import solve_ivp
import os
import json
import time


#Cleaning the terminal 
os.system('cls')


start = time.perf_counter()

#--------------------------------------------------- LOADING THE MATRICES ----------------------------------------------#

# Reading the Json file
with open('Dynamical_Matrices_Preloadedsprings_dampers_1Arm_BusFixed_simplified.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)

with open('Kinematical_Matrices_1Arm_BusFixed_OG.json', 'r') as json_file:
    data_Mat_Cin = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn_simplified']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn_simplified']))

# Reconstructing the Kinematical matrices
COEF_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['COEF_Kin']['COEF_Kin']))
RHS_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['RHS_Kin']['RHS_Kin']))



#------------------------------------------ SYMBOLIC VARIABLES ----------------------------------------------------#    

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


# Deployment spring dampers
# Deployment angles 
# Robotic arm 1
q4_deploy = (-30)*(np.pi/180)
q7_deploy = (-120)*(np.pi/180)
q10_deploy = (-60)*(np.pi/180)
q13_deploy = (-15)*(np.pi/180)

# Robotic arm 2
q18_deploy = (30)*(np.pi/180)
q21_deploy = (120)*(np.pi/180)
q24_deploy = (60)*(np.pi/180)
q27_deploy = (15)*(np.pi/180)

# Robotic arm 3
q28_deploy = (30)*(np.pi/180)
q31_deploy = (120)*(np.pi/180)
q34_deploy = (60)*(np.pi/180)
q37_deploy = (15)*(np.pi/180)

# Robotic arm 4
q42_deploy = (-30)*(np.pi/180)
q45_deploy = (-120)*(np.pi/180)
q48_deploy = (-60)*(np.pi/180)
q51_deploy = (-15)*(np.pi/180)


# ------------------------------------------------------------ #
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
    k11 : 0.5,
    k21 : 0.3,
    k31 : 0.2,
    k41 : 0.1,
    k12 : 0.5,
    k22 : 0.3,
    k32 : 0.2,
    k42 : 0.1,
    k13 : 0.5,
    k23 : 0.3,
    k33 : 0.2,
    k43 : 0.1,
    k14 : 0.5,
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


    # # ------------------------------ #
    # # Force and Torques timed application
    
    # # Torque applied for 1s<t<2.5s
    # if t > 1 and t < 2:
    #     # Robotic arm 1
    #     T11 = 0.05
    #     T21 = 0.01
    #     T31 = 0.025
    #     T41 = 0.005

    # else:
    #     # Robotic arm 1
    #     T11 = 0.0
    #     T21 = 0.0
    #     T31 = 0.0
    #     T41 = 0.0

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

    # # Robotic Arm 1
    # if q4 < (-45)*(np.pi/180):    # q4 angle saturation condition
    #     State_Vector_dot[0] = 0   # blocking the acceleration of B11 
    #     State_Vector_dot[4] = 0   # blocking the velocity of B11 
        
    # if q7 < (-105)*(np.pi/180):   # q7 angle saturation condition
    #     State_Vector_dot[1] = 0   # blocking the acceleration of B21 
    #     State_Vector_dot[5] = 0   # blocking the angle of B21

    # if q10 < (-75)*(np.pi/180):   # q10 angle saturation condition
    #     State_Vector_dot[2] = 0   # blocking the acceleration of B31 
    #     State_Vector_dot[6] = 0   # blocking the angle of B31

    # if q13 < (-15)*(np.pi/180):   # q13 angle saturation condition
    #     State_Vector_dot[3] = 0   # blocking the acceleration of B41 
    #     State_Vector_dot[7] = 0   # blocking the angle of B41


    # ------------------------------ #
    # Return the derivatives of all variables
    print(t)

    return State_Vector_dot.flatten()


# Initial conditions
    #  u4  u7   u10  u13   q4  q7   q10  q13  
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  


# Time interval for solution
t_span = (0.0, 10.0) 
t_eval = np.linspace(t_span[0], t_span[1], 1000) 

# Numerical resolution of the system with solve_ivp
solution = solve_ivp(system, t_span, y0, t_eval=t_eval, vectorized=False, method='RK23', rtol=1e-4, atol=1e-5, dense_output=True, max_step=0.3)



# -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #
# Json Python necessities

data_to_save = {
    "t": solution.t.tolist(),  # Converting the numpy array in a list
    "y": solution.y.tolist()   # Converting the numpy array in a list
}

# Saving the datas in a json file
with open('solution_1arm_Preloadedsprings_dampers_BusFixed_simplified.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

print("Dati salvati con successo in solution.json")

# ---------------------------------------------------------------------------------------------------------------- #


end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')
