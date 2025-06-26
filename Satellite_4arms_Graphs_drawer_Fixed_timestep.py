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

print('Ciaoo875')


#--------------------------------------------------- LOADING THE MATRICES ----------------------------------------------#

# Reading the Json file
with open('solution_4arms_Preloadedsprings_dampers_Fixed_step.json', 'r') as json_file:
    data = json.load(json_file)




# --------------------------------------------------------------- #
# Results extraction

# Time
t_data = np.array(data['t'])


y_data = np.array(data['y']) 

# Spacecraft bus angular displacement
u1_data = y_data[:,0]
u2_data = y_data[:,1]
u3_data = y_data[:,2]

q1_data = y_data[:,22]
q2_data = y_data[:,23]
q3_data = y_data[:,24]

# Spacecraft bus linear displacement
u52_data = y_data[:,19]
u53_data = y_data[:,20]
u54_data = y_data[:,21]

q52_data = y_data[:,41]
q53_data = y_data[:,42]
q54_data = y_data[:,43]

# Robotic arm 1
u4_data = y_data[:,3]
u7_data = y_data[:,4]
u10_data = y_data[:,5]
u13_data = y_data[:,6]

q4_data = y_data[:,25]
q7_data = y_data[:,26]
q10_data = y_data[:,27]
q13_data = y_data[:,28]

# Robotic arm 2
u18_data = y_data[:,7]
u21_data = y_data[:,8]
u24_data = y_data[:,9]
u27_data = y_data[:,10]

q18_data = y_data[:,29]
q21_data = y_data[:,30]
q24_data = y_data[:,31]
q27_data = y_data[:,32]

# Robotic arm 3
u28_data = y_data[:,11]
u31_data = y_data[:,12]
u34_data = y_data[:,13]
u37_data = y_data[:,14]

q28_data = y_data[:,33]
q31_data = y_data[:,34]
q34_data = y_data[:,35]
q37_data = y_data[:,36]

# Robotic arm 4
u42_data = y_data[:,15]
u45_data = y_data[:,16]
u48_data = y_data[:,17]
u51_data = y_data[:,18]

q42_data = y_data[:,37]
q45_data = y_data[:,38]
q48_data = y_data[:,39]
q51_data = y_data[:,40]

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
plt.plot(t_data, q2_data*(180/np.pi), label="B0 x angular displacement",color='r')
plt.xlabel("t")
plt.ylabel("x angular displacement [deg]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_data, q3_data*(180/np.pi), label="B0 y angular displacement",color='g')
plt.xlabel("t")
plt.ylabel("y angular displacement [deg]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_data, q1_data*(180/np.pi), label="B0 z angular displacement",color='b')
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

# ------------------------ #
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

plt.tight_layout()

# Robotic Arm 2
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(t_data, u18_data*(180/np.pi), label="B12 body deploying velocity",color='r')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, u21_data*(180/np.pi), label="B22 body deploying velocity", color='g')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, u24_data*(180/np.pi), label="B32 body deploying velocity", color='b')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, u27_data*(180/np.pi), label="B42 body deploying velocity", color='k')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# ------------------------ #
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(t_data, q18_data*(180/np.pi), label="B12 body deploying angle",color='r')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, q21_data*(180/np.pi), label="B22 body deploying angle", color='g')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, q24_data*(180/np.pi), label="B32 body deploying angle", color='b')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, q27_data*(180/np.pi), label="B42 body deploying angle", color='k')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# Robotic Arm 3
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(t_data, u28_data*(180/np.pi), label="B13 body deploying velocity",color='r')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, u31_data*(180/np.pi), label="B23 body deploying velocity", color='g')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, u34_data*(180/np.pi), label="B33 body deploying velocity", color='b')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, u37_data*(180/np.pi), label="B43 body deploying velocity", color='k')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# ------------------------ #
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(t_data, q28_data*(180/np.pi), label="B13 body deploying angle",color='r')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, q31_data*(180/np.pi), label="B23 body deploying angle", color='g')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, q34_data*(180/np.pi), label="B33 body deploying angle", color='b')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, q37_data*(180/np.pi), label="B43 body deploying angle", color='k')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# Robotic Arm 4
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(t_data, u42_data*(180/np.pi), label="B14 body deploying velocity",color='r')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, u45_data*(180/np.pi), label="B24 body deploying velocity", color='g')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, u48_data*(180/np.pi), label="B34 body deploying velocity", color='b')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, u51_data*(180/np.pi), label="B44 body deploying velocity", color='k')
plt.xlabel("t")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# ------------------------ #
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(t_data, q42_data*(180/np.pi), label="B14 body deploying angle",color='r')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, q45_data*(180/np.pi), label="B24 body deploying angle", color='g')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, q48_data*(180/np.pi), label="B34 body deploying angle", color='b')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, q51_data*(180/np.pi), label="B44 body deploying angle", color='k')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.show()

# ---------------------------------------------------------------------------------------------------------------- #


print('Codice terminato')
