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

print('Ciao 918')


#--------------------------------------------------- LOADING THE MATRICES ----------------------------------------------#

# Reading the Json file
with open('solution_4arms_Preloadedsprings_dampers_RotatingSc.json', 'r') as json_file:
    data = json.load(json_file)




# --------------------------------------------------------------- #
# Results extraction

# Time
t_data = np.array(data['t'])

# Spacecraft bus angular displacement
u1_data = np.array(data['y'][0])
u2_data = np.array(data['y'][1])
u3_data = np.array(data['y'][2])

q1_data = np.array(data['y'][22])
q2_data = np.array(data['y'][23])
q3_data = np.array(data['y'][24])

# Spacecraft bus linear displacement
u52_data = np.array(data['y'][19])
u53_data = np.array(data['y'][20])
u54_data = np.array(data['y'][21])

q52_data = np.array(data['y'][41])
q53_data = np.array(data['y'][42])
q54_data = np.array(data['y'][43])

# Robotic arm 1
u4_data = np.array(data['y'][3])
u7_data = np.array(data['y'][4])
u10_data = np.array(data['y'][5])
u13_data = np.array(data['y'][6])

q4_data = np.array(data['y'][25])
q7_data = np.array(data['y'][26])
q10_data = np.array(data['y'][27])
q13_data = np.array(data['y'][28])

# Robotic arm 2
u18_data = np.array(data['y'][7])
u21_data = np.array(data['y'][8])
u24_data = np.array(data['y'][9])
u27_data = np.array(data['y'][10])

q18_data = np.array(data['y'][29])
q21_data = np.array(data['y'][30])
q24_data = np.array(data['y'][31])
q27_data = np.array(data['y'][32])

# Robotic arm 3
u28_data = np.array(data['y'][11])
u31_data = np.array(data['y'][12])
u34_data = np.array(data['y'][13])
u37_data = np.array(data['y'][14])

q28_data = np.array(data['y'][33])
q31_data = np.array(data['y'][34])
q34_data = np.array(data['y'][35])
q37_data = np.array(data['y'][36])

# Robotic arm 4
u42_data = np.array(data['y'][15])
u45_data = np.array(data['y'][16])
u48_data = np.array(data['y'][17])
u51_data = np.array(data['y'][18])

q42_data = np.array(data['y'][37])
q45_data = np.array(data['y'][38])
q48_data = np.array(data['y'][39])
q51_data = np.array(data['y'][40])

# --------------------------------------------------------------- #



# ---------------------------------------- SOLUTIONS GRAPHS ------------------------------------ #

# Spacecraft bus rotational displacement
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t_data, u1_data*(180/np.pi), label="B0 x-rot. velocity",color='r', linewidth=2.0)
plt.xlabel("t [s]", fontsize=11)
plt.ylabel("Angular velocity [deg/s]", fontsize=11)
plt.grid(True)
plt.xlim(0,10)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_data, u2_data*(180/np.pi), label="B0 y-rot. velocity",color='g', linewidth=2.0)
plt.xlabel("t [s]", fontsize=11)
plt.ylabel("Angular velocity [deg/s]", fontsize=11)
plt.grid(True)
plt.xlim(0,10)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_data, u3_data*(180/np.pi), label="B0 z-rot velocity",color='b', linewidth=2.0)
plt.xlabel("t [s]", fontsize=11)
plt.ylabel("Angular velocity [deg/s]", fontsize=11)
plt.grid(True)
plt.xlim(0,10)
plt.legend()

plt.tight_layout()

# ------------------------ #
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t_data, q2_data*(180/np.pi), label="B0 x-angular displacement",color='r')
plt.xlabel("t [s]")
plt.ylabel("Angular displacement [deg]")
plt.grid(True)
plt.xlim(0,10)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_data, q3_data*(180/np.pi), label="B0 y-angular displacement",color='g')
plt.xlabel("t [s]")
plt.ylabel("Angular displacement [deg]")
plt.grid(True)
plt.xlim(0,10)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_data, q1_data*(180/np.pi), label="B0 z-angular displacement",color='b')
plt.xlabel("t [s]")
plt.ylabel("Angular displacement [deg]")
plt.grid(True)
plt.xlim(0,10)
plt.legend()

plt.tight_layout()


# Spacecraft bus linear displacement
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.xlim(0,10)
plt.plot(t_data, u52_data, label="Spacecraft bus x-linear velocity",color='r')
plt.xlabel("t [s]")
plt.ylabel("velocity [m/s]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.xlim(0,10)
plt.plot(t_data, u53_data, label="Spacecraft bus y-linear velocity",color='g')
plt.xlabel("t [s]")
plt.ylabel("velocity [m/s]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.xlim(0,10)
plt.plot(t_data, u54_data, label="Spacecraft bus z-linear velocity",color='b')
plt.xlabel("t [s]")
plt.ylabel("velocity [m/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# ------------------------ #
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.xlim(0,10)
plt.plot(t_data, q52_data, label="Spacecraft bus x-linear displacement",color='r')
plt.xlabel("t [s]")
plt.ylabel("lenght [m]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.xlim(0,10)
plt.plot(t_data, q53_data, label="Spacecraft bus y-linear displacement",color='g')
plt.xlabel("t [s]")
plt.ylabel("lenght [m]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.xlim(0,10)
plt.plot(t_data, q54_data, label="Spacecraft bus z-linear displacement",color='b')
plt.xlabel("t [s]")
plt.ylabel("lenght [m]")
plt.grid(True)
plt.legend()

plt.tight_layout()


# Robotic Arm 1
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.xlim(0,10)
plt.plot(t_data, u4_data*(180/np.pi), label="B11 body deploying velocity",color='#456773')
plt.xlabel("t [s]")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.xlim(0,10)
plt.plot(t_data, u7_data*(180/np.pi), label="B21 body deploying velocity", color='#54793e')
plt.xlabel("t [s]")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.xlim(0,10)
plt.plot(t_data, u10_data*(180/np.pi), label="B31 body deploying velocity", color='#d46d0c')
plt.xlabel("t [s]")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.xlim(0,10)
plt.plot(t_data, u13_data*(180/np.pi), label="B41 body deploying velocity", color='#7a0c00')
plt.xlabel("t [s]")
plt.ylabel("Deploying velocity [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# ------------------------ #
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.xlim(0,10)
plt.plot(t_data, q4_data*(180/np.pi), label="B11 body deploying angle",color='#5c8a99', linewidth=2.0)
plt.axhline(y=-30, color='#5c8a99', linestyle='--', linewidth=1.7, label="Target deploy angle = -30°")
plt.xlabel("t [s]", fontsize=11)
plt.ylabel("Deploying angle [deg]", fontsize=11)
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.xlim(0,10)
plt.plot(t_data, q7_data*(180/np.pi), label="B21 body deploying angle", color='#6a994e', linewidth=2.0)
plt.axhline(y=-120, color='#6a994e', linestyle='--', linewidth=1.7, label="Target deploy angle = -120°")
plt.xlabel("t [s]", fontsize=11)
plt.ylabel("Deploying angle [deg]", fontsize=11)
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.xlim(0,10)
plt.plot(t_data, q10_data*(180/np.pi), label="B31 body deploying angle", color='#f38b2b', linewidth=2.0)
plt.axhline(y=-60, color='#f38b2b', linestyle='--', linewidth=1.7, label="Target deploy angle = -60°")
plt.xlabel("t [s]", fontsize=11)
plt.ylabel("Deploying angle [deg]", fontsize=11)
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.xlim(0,10)
plt.plot(t_data, q13_data*(180/np.pi), label="B41 body deploying angle", color='#ba1200', linewidth=2.0)
plt.axhline(y=-15, color='#ba1200', linestyle='--', linewidth=1.7, label="Target deploy angle = -15°")
plt.xlabel("t [s]", fontsize=11)
plt.ylabel("Deploying angle [deg]", fontsize=11)
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
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, q21_data*(180/np.pi), label="B22 body deploying angle", color='g')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, q24_data*(180/np.pi), label="B32 body deploying angle", color='b')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, q27_data*(180/np.pi), label="B42 body deploying angle", color='k')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg]")
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
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, q31_data*(180/np.pi), label="B23 body deploying angle", color='g')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, q34_data*(180/np.pi), label="B33 body deploying angle", color='b')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, q37_data*(180/np.pi), label="B43 body deploying angle", color='k')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg]")
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
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_data, q45_data*(180/np.pi), label="B24 body deploying angle", color='g')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_data, q48_data*(180/np.pi), label="B34 body deploying angle", color='b')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_data, q51_data*(180/np.pi), label="B44 body deploying angle", color='k')
plt.xlabel("t")
plt.ylabel("Deploying angle [deg]")
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.show()

# ---------------------------------------------------------------------------------------------------------------- #


print('Codice terminato')
