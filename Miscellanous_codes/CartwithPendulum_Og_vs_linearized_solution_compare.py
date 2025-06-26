# Codice per plottare sullo stesso graifico i risultati della dinamica del moto del "Cart with Pendulum" nel caso di 
# matrici originali e matrici linearizzate.


# ------------------------------------ IMPORTING THE NECESSARY PACKAGES ----------------------------------------------- #

# Importing only the system package to tell python in which other folders search the other packages
import sys
import os

# To find the files with the functions, we need to tell python to "go up two folders" from the folder in which this file is 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing the other packages
from utility.import_packages import sy, time, json, solve_ivp, plt, np

# Importing the functions
from utility.numerical_solver_functions import Matrix_Numerical_Integration
from utility.miscellaneous_functions import load_matrices_from_json


# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

#Cleaning the terminal 
os.system('cls')


# ---------------------------------------------- READING THE FILES ------------------------------------------------ #

# Reading the Json file
with open('Numerical_Solver/Cart_with_Pendulum/solution_Cart_with_Pendulum_45.json', 'r') as json_file:
    data = json.load(json_file)

with open('Numerical_Solver/Cart_with_Pendulum/solution_Cart_with_Pendulum_linearized_45.json', 'r') as json_file:
    data_lin = json.load(json_file)



# ------------------------------------------------ GRAFICARE LA SOLUZIONE -------------------------------------------- #

# Estrazione dei risultati
t_data = np.array(data['t']) 
t_data_lin = np.array(data['t']) 

# Normal
u1_solution = np.array(data['y'][0])
u2_solution = np.array(data['y'][1])
q1_solution = np.array(data['y'][2])
q2_solution = np.array(data['y'][3])

#Linearized
u1_lin_solution = np.array(data_lin['y'][0])
u2_lin_solution = np.array(data_lin['y'][1])
q1_lin_solution = np.array(data_lin['y'][2])
q2_lin_solution = np.array(data_lin['y'][3])


# Grafico della soluzione
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_data, u1_solution*(180/np.pi), label="Original",color='#5C8a99')
plt.plot(t_data_lin, u1_lin_solution*(180/np.pi), label="Linearized", color='#6a994e')
plt.xlabel("t [s]")
plt.ylabel("Pendulum - angular velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_data, u2_solution, label="Original", color='#f38b2b')
plt.plot(t_data_lin, u2_lin_solution, label="Linearized", color='#ba1200')
plt.xlabel("t [s]")
plt.ylabel("Cart - linear velocity [m/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_data, q1_solution*(180/np.pi), label="Original",color='#5C8a99')
plt.plot(t_data_lin, q1_lin_solution*(180/np.pi), label="Linearized", color='#6a994e')
plt.xlabel("t [s]")
plt.ylabel("Pendulum - angular displacement [deg]")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_data, q2_solution, label="Original", color='#f38b2b')
plt.plot(t_data_lin, q2_lin_solution, label="Linearized", color='#ba1200')
plt.xlabel("t [s]")
plt.ylabel("Cart - linear displacement [m]")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()