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


#Cleaning the terminal 
os.system('cls')


'''
#--------------------------------------------------- PRODUCING SOME MATRICES ----------------------------------------------#
# Produciamo due matrici di test, A è una 2x2 e l'altra è il vettore b 2x1

# Definizione simbolica delle matrici
t = sy.symbols('t')
q1 = me.dynamicsymbols('q1')
q2 = me.dynamicsymbols('q2')
u1 = me.dynamicsymbols('u1')
u2 = me.dynamicsymbols('u2')
k = sy.symbols('k')
bc = sy.symbols('b')
m1 = sy.symbols('m1')
m2 = sy.symbols('m2')
l = sy.symbols('l')
g = sy.symbols('g')



# Matrici Dinamiche
COEF = sy.Matrix([[(m1+m2), m2*(l/2)*sy.cos(q2)], 
                  [m2*(l/2)*sy.cos(q2), (1/3)*m2*(l**2)]
                 ])
RHS = sy.Matrix([(m2/2)*l*(u2**2)*sy.sin(q2) - k*q1 - bc*u1, -g*m2*(l/2)*sy.sin(q2)])
StVec = sy.Matrix([u1.diff(t), u2.diff(t)])

# Matrici Cinematiche
A = sy.Matrix([[1, 0],
               [0, 1]
               ])

b = sy.Matrix([u1, u2])

qdot = sy.Matrix([q1.diff(t), q2.diff(t)])

# Converti le matrici in dizionari (serializzabili in JSON)
# Matrici Dinamiche
COEF_dict = {'COEF': str(COEF)}
RHS_dict = {'RHS': str(RHS)}
StVec_dict = {'StVec': str(StVec)}

# Matrici Cinematiche
A_dict = {'A': str(A)}
b_dict = {'b': str(b)}
qdot_dict = {'qdot': str(qdot)}



# Scrivi i dizionari su file JSON
# Matrici Dinamiche
with open('Matrici_Dinamiche.json', 'w') as json_file:
    json.dump({'COEF': COEF_dict, 'RHS': RHS_dict, 'StVec': StVec_dict}, json_file)

# Matrici Cinematiche
with open('Matrici_Cinematiche.json', 'w') as json_file:
    json.dump({'A': A_dict, 'b': b_dict, 'qdot': qdot_dict}, json_file)


'''
#--------------------------------------------------- LOADING THE MATRICES ----------------------------------------------#

# Reading the Json file
with open('Dynamical_Matrices.json', 'r') as json_file:
    data_Mat_Din = json.load(json_file)

with open('Kinematical_Matrices.json', 'r') as json_file:
    data_Mat_Cin = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Din['COEF_Dyn']['COEF_Dyn']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Din['RHS_Dyn']['RHS_Dyn']))

# Reconstructing the Kinematical matrices
COEF_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['COEF_Kin']['COEF_Kin']))
RHS_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['RHS_Kin']['RHS_Kin']))

# Loading the generalized coordinates vector
q_vect_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['q_vector']['q_vector']))


'''
# --------------------------------------------------- VARIABLES DEFINITIONS ---------------------------------------------- #

# We need to look into the matrices we loaded to search for the symbolic variables and the gen.speeds and coordinates.
# We have to do that to define these things as symbolic variables before doing the calculations
 
# Dynamical matrices: Symbolic Variables 
COEF_Dyn_var =  COEF_Dyn_loaded.free_symbols
RHS_Dyn_var =  RHS_Dyn_loaded.free_symbols

# Now we search for the time varing variables, our gen. speeds and gen. coordinates that are stored inside the "RHS_Kin_loaded"
# and "q_vect_loaded vectors" 
gen_speeds = me.find_dynamicsymbols(RHS_Kin_loaded)
gen_coord = me.find_dynamicsymbols(q_vect_loaded)


# Now we merge all the symbolic variables togheter and time varing variables togheter
all_symbolic_var = COEF_Dyn_var.union(RHS_Dyn_var)
all_time_varing_var = gen_speeds.union(gen_coord)



# And at last, we initialize all these variables we have extrapolated from our matrices
init_symbolic_var = {s: sy.symbols(str(s)) for s in all_symbolic_var}
init_time_varing_var = {s: me.dynamicsymbols(str(s)) for s in all_time_varing_var}

# Now we print these two vectors, to see which variables we need to define
sy.pprint(all_symbolic_var)
sy.pprint(all_time_varing_var)

'''
# --------------------------------------------------------------- #


# Initializing the variables
t = sy.symbols('t')                 # Definition of the time variable
l = sy.symbols('l')                 # Spacecraft bus length [m]
w = sy.symbols('w')                 # Spacecraft bus width [m]
h = sy.symbols('h')                 # Spacecraft bus height [m]
d = sy.symbols('d')                 # Spacecraft bus-arm joint distance [m]
bl = sy.symbols('bl')               # Robotic arm long piece length [m]
hl = sy.symbols('hl')               # Robotic arm long piece height [m]
wl = sy.symbols('wl')               # Robotic arm long piece width [m]
bs = sy.symbols('bs')               # Robotic arm short piece length [m]
hs = sy.symbols('hs')               # Robotic arm short piece height [m]
ws = sy.symbols('ws')               # Robotic arm short piece width [m]
T = sy.symbols('T')                 # Torque acting on the arms [N*m]
mlongarm = sy.symbols('mlongarm')   # Robotic Long arm piece mass [kg]
mshortarm = sy.symbols('mshortarm') # Robotic Short arm piece mass [kg]
mbus = sy.symbols('mbus')           # Spacecraft bus mass [kg]


# Initializing the gen.coordinates and speeds
q1, q2, q3, q4, q5, q6, q7, q10, q13, q16 = sy.symbols('q1 q2 q3 q4 q5 q6 q7 q10 q13 q16')
u1, u2, u3, u4, u5, u6, u7, u10, u13, u16 = sy.symbols('u1 u2 u3 u4 u5 u6 u7 u10 u13 u16')



# ----------------------------------------- SOLVING THE SYSTEM ---------------------------------------------------- #

# Now we have to create a dictionary for the symbolic variables substituition inside the matrices. Practically we need
# to specify the masses, lenghts, ..., values for the system parameters

# Dictionary
substitution = {
    l: 0.6,                 # Spacecraft bus length [m]
    w: 0.5,                 # Spacecraft bus width [m]
    h: 0.5,                 # Spacecraft bus height [m]
    d: 0.1,                 # Spacecraft bus-arm joint distance [m]
    bl: 0.25,               # Robotic arm long piece length [m]
    hl: 0.05,               # Robotic arm long piece height [m]
    wl: 0.05,               # Robotic arm long piece width [m]
    bs: 0.16,               # Robotic arm short piece length [m]
    hs: 0.05,               # Robotic arm short piece height [m]
    ws: 0.05,               # Robotic arm short piece width [m]
    T: 5,                   # Torque acting on the arms [N*m]
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
Mass_Matrix = sy.BlockMatrix([[COEF_Dyn_loaded, sy.zeros(*COEF_Kin_loaded.shape)],
                              [sy.zeros(*COEF_Dyn_loaded.shape), COEF_Kin_loaded]])
Forcing_Vector = sy.Matrix.vstack(RHS_Dyn_loaded, RHS_Kin_loaded)


# # Translating the symbolic matrices from sympy to numpy
# all_symbolic_var = (u1,u2,u3,u4,u5,u6,u7,u10,u13,u16,q1,q2,q3,q4,q5,q6,q7,q10,q13,q16)
# Mass_Matrix_lambdified = sy.lambdify(all_symbolic_var,Mass_Matrix, modules ='numpy')
# Forcing_Vector_lambdified = sy.lambdify(all_symbolic_var,Forcing_Vector, modules ='numpy')

def system(t, State_Vector):

    print('spacchettooo')
    # Unpacking the results
    unpack = {
        u1 : State_Vector[0],
        u2 : State_Vector[1],
        u3 : State_Vector[2],
        u4 : State_Vector[3],
        u5 : State_Vector[4],
        u6 : State_Vector[5],
        u7 : State_Vector[6],
        u10 : State_Vector[7],
        u13 : State_Vector[8],
        u16 : State_Vector[9],

        q1 : State_Vector[10],
        q2 : State_Vector[11],
        q3 : State_Vector[12],
        q4 : State_Vector[13],
        q5 : State_Vector[14],
        q6 : State_Vector[15],
        q7 : State_Vector[16], 
        q10 : State_Vector[17],
        q13 : State_Vector[18], 
        q16 : State_Vector[19],

        t : t,
    }
    
    
    # Sostituisci le variabili simboliche con i valori numerici in Mass_Matrix e Forcing_Vector
    print('sostituiscoo')
    # Mass_Matrix_eval = Mass_Matrix_lambdified(u1,u2,u3,u4,u5,u6,u7,u10,u13,u16,q1,q2,q3,q4,q5,q6,q7,q10,q13,q16)
    # Forcing_Vector_eval = Forcing_Vector_lambdified(u1,u2,u3,u4,u5,u6,u7,u10,u13,u16,q1,q2,q3,q4,q5,q6,q7,q10,q13,q16)
    Mass_Matrix_eval = Mass_Matrix.xreplace(unpack)
    Forcing_Vector_eval = Forcing_Vector.xreplace(unpack)

    print('valutoo')
    Mass_Matrix_eval = Mass_Matrix_eval.evalf()
    Forcing_Vector_eval = Forcing_Vector_eval.evalf()

    print('convertoo')
    # Convertire in un array numpy per il calcolo
    Mass_Matrix_numeric = np.array(Mass_Matrix_eval).astype(np.float64)
    Forcing_Vector_numeric = np.array(Forcing_Vector_eval).astype(np.float64)

    # Calcolo dello state vector
    print('invertoo')
    try:
        
        # Risolvere il sistema lineare senza calcolare esplicitamente l'inversa
        State_Vector_dot = np.linalg.solve(Mass_Matrix_numeric, Forcing_Vector_numeric)
        print('diretta')
    except np.linalg.LinAlgError:
        
        # Se la matrice è singolare, puoi usare la pseudo-inversa
        State_Vector_dot = np.linalg.pinv(Mass_Matrix_numeric) @ Forcing_Vector_numeric
        print('pseudoinversa')

    # Restituire le derivate di tutte le variabili
    print(t)
    return State_Vector_dot.flatten()

# Condizioni iniziali
      #u1,  u2,  u3,  u4,  u5,  u6,  u7, u10, u13, u16,  q1,  q2,  q3,  q4,  q5,  q6,  q7, q10, q13, q16 
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  


# Intervallo di tempo per la soluzione
t_span = (0, 3)  # Risolvi dal tempo 0 al tempo 10
t_eval = np.linspace(t_span[0], t_span[1], 100)  # 100 punti di valutazione

# Risoluzione numerica del sistema con solve_ivp
solution = solve_ivp(system, t_span, y0, t_eval=t_eval, vectorized=False, method='RK45', rtol=1e-2, atol=1e-4)

# Estrazione dei risultati
u1_solution = solution.y[0]
u2_solution = solution.y[1]
q1_solution = solution.y[2]
q2_solution = solution.y[3]

# ---------------------------------------- GRAFICARE LA SOLUZIONE ------------------------------------ #
# Grafico della soluzione
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(solution.t, u1_solution, label="u1 (velocità)")
plt.xlabel("Tempo")
plt.ylabel("u1 (velocità)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(solution.t, u2_solution*(180/np.pi), label="u2 (velocità)", color='r')
plt.xlabel("Tempo")
plt.ylabel("u2 (velocità)")
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(solution.t, q1_solution, label="x1 (posizione)")
plt.xlabel("Tempo")
plt.ylabel("x1 (posizione)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(solution.t, q2_solution*(180/np.pi), label="x2 (posizione)", color='r')
plt.xlabel("Tempo")
plt.ylabel("x2 (posizione)")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()



print('Codice terminato')