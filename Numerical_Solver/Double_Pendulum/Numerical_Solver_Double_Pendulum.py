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
from matplotlib.animation import FuncAnimation
import time

#Cleaning the terminal 
os.system('cls')


# ---------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

start = time.perf_counter()


#--------------------------------------------------- LOADING THE MATRICES ----------------------------------------------#

# Reading the Json file
with open('Matrix_Kane/Kane_MatrixForm_Double_Pendulum/Dynamical_Matrices_DoublePendulum.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)

with open('Matrix_Kane/Kane_MatrixForm_Double_Pendulum/Kinematical_Matrices_DoublePendulum.json', 'r') as json_file:
    data_Mat_Cin = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']))

# Reconstructing the Kinematical matrices
COEF_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['COEF_Kin']['COEF_Kin']))
RHS_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['RHS_Kin']['RHS_Kin']))


# --------------------------------------------------------------- #

# Initializing the variables
# Initial motion parameters
t = sy.symbols('t')                # Definition of the time variable
g = sy.symbols('g')                # Gravitational acceleration [m/s^2]
m = sy.symbols('m')                # Particles Mass [kg]
l = sy.symbols('l')                # Rod length [m]
T = sy.symbols('T')                # Torque applied
F = sy.symbols('F')                # Force applied



# Initializing the gen.coordinates and speeds
q6, q9 = sy.symbols('q6 q9')
u6, u9  = sy.symbols('u6 u9')

# # Initializing the perturbed gen.coordinates and speeds
# dq6, dq9 = sy.symbols('dq6 dq9')
# du6, du9  = sy.symbols('du6 du9')



# ----------------------------------------- SOLVING THE SYSTEM ---------------------------------------------------- #

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

#--------------------------------------------------- CALCULATIONS -------------------------------------------------------#

# We assemble the giant mass matrix for the system we want to solve, its composed by the mass matrix of the dynamical and 
# kinematicl equations. The same we do to assemble the giant forcing vector
Mass_Matrix = sy.BlockDiagMatrix(COEF_Dyn_loaded, COEF_Kin_loaded)
Forcing_Vector = sy.Matrix.vstack(RHS_Dyn_loaded, RHS_Kin_loaded)


# Transforming the matrices from "blockmatrices" to "normal matrices"
Mass_Matrix_dense = Mass_Matrix.as_mutable()
Forcing_Vector_dense = Forcing_Vector.as_mutable()

# Translating the symbolic matrices from sympy to numpy
all_symbolic_var = (u6,u9,q6,q9)
# all_symbolic_var = (du6,du9,dq6,dq9)
Mass_Matrix_lambdified = sy.lambdify(all_symbolic_var, Mass_Matrix_dense, 'numpy')
Forcing_Vector_lambdified = sy.lambdify(all_symbolic_var, Forcing_Vector_dense,'numpy')


def system(t, State_Vector):

    # Unpacking
    u6 = State_Vector[0]
    u9 = State_Vector[1]

    q6 = State_Vector[2]
    q9 = State_Vector[3]

    t = t

    # Sostituisci le variabili simboliche con i valori numerici in Mass_Matrix e Forcing_Vector
    Mass_Matrix_eval = Mass_Matrix_lambdified(u6,u9,q6,q9)
    Forcing_Vector_eval = Forcing_Vector_lambdified(u6,u9,q6,q9)
   
    try:
        
        # Risolvere il sistema lineare senza calcolare esplicitamente l'inversa
        State_Vector_dot = np.linalg.solve(Mass_Matrix_eval, Forcing_Vector_eval)
        
    except np.linalg.LinAlgError:
        
        # Se la matrice è singolare, puoi usare la pseudo-inversa
        State_Vector_dot = np.dot(np.linalg.pinv(Mass_Matrix_eval), Forcing_Vector_eval)
      
    # Restituire le derivate di tutte le variabili
    print(t)

    return State_Vector_dot.flatten()

# Condizioni iniziali
      #u6,  u9,  q6,  q9 
y0 = [0.0, 0.0, 0.0, np.pi/4]


# Intervallo di tempo per la soluzione
t_span = (0.0, 10)  # Risolvi dal tempo 0 al tempo 10
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # 1000 punti di valutazione

# Risoluzione numerica del sistema con solve_ivp
solution = solve_ivp(system, t_span, y0, t_eval=t_eval, vectorized=False, method='RK45')


# Estrazione dei risultati
u6_solution = solution.y[0]
u9_solution = solution.y[1]
q6_solution = solution.y[2]
q9_solution = solution.y[3]



# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

end = time.perf_counter()



# ---------------------------------------- GRAFICARE LA SOLUZIONE ------------------------------------ #
# Grafico della soluzione
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(solution.t, u6_solution*(180/np.pi), label="x_dot (velocità)")
plt.xlabel("Tempo")
plt.ylabel("u1 (velocità)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(solution.t, u9_solution*(180/np.pi), label="teta_dot (velocità)", color='r')
plt.xlabel("Tempo")
plt.ylabel("u2 (velocità)")
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(solution.t, q6_solution*(180/np.pi), label="x (posizione)")
plt.xlabel("Tempo")
plt.ylabel("x1 (posizione)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(solution.t, q9_solution*(180/np.pi), label="teta (posizione)", color='r')
plt.xlabel("Tempo")
plt.ylabel("x2 (posizione)")
plt.grid(True)
plt.legend()

plt.tight_layout()

# Showing the animation into a figure
plt.show()


# solution_str = {'solution': str(solution)}

# with open('solution.json', 'w') as json_file:
#     json.dump({'solution': solution_str},json_file)

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print('Codice terminato')
