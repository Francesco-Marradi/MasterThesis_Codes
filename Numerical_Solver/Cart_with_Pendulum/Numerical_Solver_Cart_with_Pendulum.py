#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI-----------------------------------------------#
#------------------------------------------- SCRIPT OBJECTIVE ---------------------------------------------------------#

# The objective of this script is to solve the eq. of motion of a system by solving numerically the linear matrix 
# system COEF = udot*RHS




# ------------------------------------ IMPORTING THE NECESSARY PACKAGES ----------------------------------------------- #

# Importing only the system package to tell python in which other folders search the other packages
import sys
import os

# To find the files with the functions, we need to tell python to "go up two folders" from the folder in which this file is 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# Importing the other packages
from utility.import_packages import sy, time, json, solve_ivp, plt, np

# Importing the functions
from utility.numerical_solver_functions import Matrix_Numerical_Integration
from utility.miscellaneous_functions import load_matrices_from_json


# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

#Cleaning the terminal 
os.system('cls')



# ---------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

start = time.perf_counter()



#--------------------------------------------------- LOADING THE MATRICES ----------------------------------------------#

# Reading the Json file
COEF_Dyn_loaded, RHS_Dyn_loaded, COEF_Kin_loaded, RHS_Kin_loaded = load_matrices_from_json('Dynamical_Matrices_Cart_with_Pendulum.json','Kinematical_Matrices_Cart_with_Pendulum.json', 'Matrix_Kane/Kane_MatrixForm_Cart_with_Pendulum')
# COEF_Dyn_loaded, RHS_Dyn_loaded, COEF_Kin_loaded, RHS_Kin_loaded = load_matrices_from_json('Dynamical_Matrices_Cart_linearized.json','Kinematical_Matrices_Cart_linearized.json', 'Matrix_Kane/Kane_MatrixForm_Cart_with_Pendulum')
# COEF_Dyn_loaded, RHS_Dyn_loaded, COEF_Kin_loaded, RHS_Kin_loaded = load_matrices_from_json('Dynamical_Matrices_Cart_with_Pendulum_simplified.json','Kinematical_Matrices_Cart_with_Pendulum.json', 'Matrix_Kane/Kane_MatrixForm_Cart_with_Pendulum')



#-------------------------------------------- SYMBOLIC VARIABLES -------------------------------------------------------#    

# Initializing the variables
t = sy.symbols('t')                # Definition of the time variable
g = sy.symbols('g')                # Gravitational acceleration [m/s^2]
m0 = sy.symbols('m0')              # Mass of body 1 [kg]
m1 = sy.symbols('m1')              # Mass of body 2 [kg]
b = sy.symbols('b')                # Damper coefficient [N*s/m]
k = sy.symbols('k')                # spring constant [N/m]
l = sy.symbols('l')                # Rod length [m]
h = sy.symbols('h')                # Box height [m]
w = sy.symbols('w')                # Box whidt [m]
a = sy.symbols('a')                # Cart center of mass's height wrt x axis [m] 



# Initializing the gen.coordinates and speeds
q6, q7 = sy.symbols('q6 q7')
u6, u7  = sy.symbols('u6 u7')

# Initializing the perturbed gen.coordinates and speeds
# dq6, dq7 = sy.symbols('dq6 dq7')
# du6, du7  = sy.symbols('du6 du7')


# Defining all the symbolic variables here, to then used them to lambdify the matrices
all_symbolic_var = (u6,u7,q6,q7)
# all_symbolic_var = (du6,du7,dq6,dq7)



# ----------------------------------------- SOLVING THE SYSTEM ---------------------------------------------------- #

# Now we have to create a dictionary for the symbolic variables substituition inside the matrices. Practically we need
# to specify the masses, lenghts, ..., values for the system parameters

# Dictionary
substitution = {
    g: 9.81,  # Accelerazione gravitazionale [m/s^2]
    m0: 1,    # Massa del carrello [kg]
    m1: 1,    # Massa del pendolo [kg]
    b: 1,     # Coefficiente smorzatore [N*s/m]
    k: 1,     # Costante della molla [N/m]
    l: 2,     # Lunghezza della sbarra [m]
}



#--------------------------------------------------- CALCULATIONS -------------------------------------------------------#

# Initial conditions
      #u6,  u7,  q6,  q7 
y0 = [0.0, 0.0, np.pi/4, 0.0]

t_span = (0.0, 10)  # Risolvi dal tempo 0 al tempo 10
n_eval = 1000       # Number of evaluation points

# Using the Integration function
Num_int_sol = Matrix_Numerical_Integration(y0, t_span, n_eval, COEF_Dyn_loaded, RHS_Dyn_loaded, COEF_Kin_loaded, RHS_Kin_loaded, substitution, all_symbolic_var, 'RK45', 1e-5, 1e-4, True)



# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

end = time.perf_counter()



# ------------------------------------------------ GRAFICARE LA SOLUZIONE -------------------------------------------- #

# Estrazione dei risultati
u6_solution = Num_int_sol.y[0]
u7_solution = Num_int_sol.y[1]
q6_solution = Num_int_sol.y[2]
q7_solution = Num_int_sol.y[3]


# Grafico della soluzione
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(Num_int_sol.t, u7_solution, label="x_dot (velocità)")
plt.xlabel("Tempo")
plt.ylabel("u1 (velocità)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Num_int_sol.t, u6_solution*(180/np.pi), label="teta_dot (velocità)", color='r')
plt.xlabel("Tempo")
plt.ylabel("u2 (velocità)")
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(Num_int_sol.t, q7_solution, label="x (posizione)")
plt.xlabel("Tempo")
plt.ylabel("x1 (posizione)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Num_int_sol.t, q6_solution*(180/np.pi), label="teta (posizione)", color='r')
plt.xlabel("Tempo")
plt.ylabel("x2 (posizione)")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()



# -------------------------------------- PYTHON NECESSITIES -------------------------------------------- #
# Json Python necessities

data_to_save = {
    "t": Num_int_sol.t.tolist(),  # Converting the numpy array in a list
    "y": Num_int_sol.y.tolist()   # Converting the numpy array in a list
}

# Saving the datas in a json file
with open('solution_Cart_with_Pendulum.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

print("Dati salvati con successo in solution.json")


# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')

