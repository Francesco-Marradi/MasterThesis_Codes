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
with open('solution_1arm_torqueapplied.json', 'r') as json_file:
    data = json.load(json_file)




# --------------------------------------------------------------- #
# Results extraction

# Time
t_data = np.array(data['t'])

# Spacecraft bus angular displacement
u1_data = np.array(data['y'][0])
u2_data = np.array(data['y'][1])
u3_data = np.array(data['y'][2])

q1_data = np.array(data['y'][10])
q2_data = np.array(data['y'][11])
q3_data = np.array(data['y'][12])

# Spacecraft bus linear displacement
u52_data = np.array(data['y'][7])
u53_data = np.array(data['y'][8])
u54_data = np.array(data['y'][9])

q52_data = np.array(data['y'][17])
q53_data = np.array(data['y'][18])
q54_data = np.array(data['y'][19])

# Robotic arm 1
u4_data = np.array(data['y'][3])
u7_data = np.array(data['y'][4])
u10_data = np.array(data['y'][5])
u13_data = np.array(data['y'][6])

q4_data = np.array(data['y'][13])
q7_data = np.array(data['y'][14])
q10_data = np.array(data['y'][15])
q13_data = np.array(data['y'][16])

# --------------------------------------------------------------- #


# ---------------------------------------- SOLUTIONS GRAPHS ------------------------------------ #

# Spacecraft bus rotational displacement
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t_data, u1_data*(180/np.pi), label="B0 x-rot. velocity",color='r')
plt.xlabel("t")
plt.ylabel("x-rot.velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_data, u2_data*(180/np.pi), label="B0 y-rot. velocity",color='g')
plt.xlabel("t")
plt.ylabel("y-rot velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_data, u3_data*(180/np.pi), label="B0 z-rot velocity",color='b')
plt.xlabel("t")
plt.ylabel("z-rot velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# ------------------------ #
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t_data, q1_data*(180/np.pi), label="B0 x angular displacement",color='r')
plt.xlabel("t")
plt.ylabel("x angular displacement [deg]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_data, q2_data*(180/np.pi), label="B0 y angular displacement",color='g')
plt.xlabel("t")
plt.ylabel("y angular displacement [deg]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_data, q3_data*(180/np.pi), label="B0 z angular displacement",color='b')
plt.xlabel("t")
plt.ylabel("z angular displacement [deg]")
plt.grid(True)
plt.legend()

plt.tight_layout()


# Spacecraft bus linear displacement
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t_data, u52_data, label="B0 x-linear velocity",color='r')
plt.xlabel("t")
plt.ylabel("x velocity [m/s]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_data, u53_data*(180/np.pi), label="B0 y-linear velocity",color='g')
plt.xlabel("t")
plt.ylabel("y linear velocity [m/s]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_data, u54_data*(180/np.pi), label="B0 z-linear velocity",color='b')
plt.xlabel("t")
plt.ylabel("z linear velocity [m/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# ------------------------ #
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t_data, q52_data, label="B0 x linear displacement",color='r')
plt.xlabel("t")
plt.ylabel("x linear displacement [m]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_data, q53_data, label="B0 y linear displacement",color='g')
plt.xlabel("t")
plt.ylabel("y linear displacement [m]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_data, q54_data, label="B0 z linear displacement",color='b')
plt.xlabel("t")
plt.ylabel("z linear displacement [m]")
plt.grid(True)
plt.legend()

plt.tight_layout()



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
