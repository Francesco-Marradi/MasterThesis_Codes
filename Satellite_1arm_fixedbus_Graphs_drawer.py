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
with open('solution_1arm_Preloadedsprings_dampers_BusFixed_OG.json', 'r') as json_file:
    data = json.load(json_file)




# --------------------------------------------------------------- #
# Results extraction

# Time
t_data = np.array(data['t'])
# Robotic arm 1
u4_data = np.array(data['y'][0])
u7_data = np.array(data['y'][1])
u10_data = np.array(data['y'][2])
u13_data = np.array(data['y'][3])

q4_data = np.array(data['y'][4])
q7_data = np.array(data['y'][5])
q10_data = np.array(data['y'][6])
q13_data = np.array(data['y'][7])

# --------------------------------------------------------------- #


# ---------------------------------------- SOLUTIONS GRAPHS ------------------------------------ #

# Robotic Arm 1
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(t_data, u4_data*(180/np.pi), label="B11 body deploying velocity",color='r')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, u7_data*(180/np.pi), label="B21 body deploying velocity", color='g')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, u10_data*(180/np.pi), label="B31 body deploying velocity", color='b')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, u13_data*(180/np.pi), label="B41 body deploying velocity", color='k')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(t_data, q4_data*(180/np.pi), label="B11 body deploying angle",color='r')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, q7_data*(180/np.pi), label="B21 body deploying angle", color='g')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, q10_data*(180/np.pi), label="B31 body deploying angle", color='b')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, q13_data*(180/np.pi), label="B41 body deploying angle", color='k')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()


plt.show()

# ---------------------------------------------------------------------------------------------------------------- #


print('Codice terminato')
