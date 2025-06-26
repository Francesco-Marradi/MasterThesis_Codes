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
with open('solution_spinningspacecraft_test.json', 'r') as json_file:
    data = json.load(json_file)

# with open('wxh_test.json', 'r') as json_file:
#     datawxh = json.load(json_file)


# Tutti i simboli usati nell'espressione
mbus, h, w, l, u1, u2, u3, q1, q2, q3 = sy.symbols('mbus h w l u1 u2 u3 q1 q2 q3')
# wxh = sy.Matrix(sy.sympify((datawxh['wxh_dict']['wxh_dict'])))
# wxh = wxh.xreplace({mbus:122.9, h:0.5 , w:0.5, l:0.6})

# --------------------------------------------------------------- #
# Results extraction

# Time
t_data = np.array(data['t'])

# Spacecraft bus angular 
u1_data = np.array(data['y'][0])
u2_data = np.array(data['y'][1])
u3_data = np.array(data['y'][2])

q1_data = np.array(data['y'][6])
q2_data = np.array(data['y'][7])
q3_data = np.array(data['y'][8])

# Spacecraft bus linear 
u52_data = np.array(data['y'][3])
u53_data = np.array(data['y'][4])
u54_data = np.array(data['y'][5])

q52_data = np.array(data['y'][9])
q53_data = np.array(data['y'][10])
q54_data = np.array(data['y'][11])


# --------------------------------------------------------------- #
# wxh_lamb = sy.lambdify((u1,u2,u3,q1,q2,q3), wxh, 'numpy' )
# wxh_eval = [0]*len(u1_data)
# wxh_evalx = [0]*len(u1_data)
# wxh_evaly = [0]*len(u1_data)
# wxh_evalz = [0]*len(u1_data)

# for i in range(len(u1_data)):
#    wxh_eval = wxh_lamb(u1_data[i],u2_data[i],u3_data[i],q1_data[i], q2_data[i], q3_data[i])
#    wxh_evalx[i] = wxh_eval[0]
#    wxh_evaly[i] = wxh_eval[1]
#    wxh_evalz[i] = wxh_eval[2]


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
plt.plot(t_data, u53_data, label="B0 y-linear velocity",color='g')
plt.xlabel("t")
plt.ylabel("y linear velocity [m/s]")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_data, u54_data, label="B0 z-linear velocity",color='b')
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


# # ------------------------ #
# plt.figure(figsize=(10, 6))
# plt.subplot(3, 1, 1)
# plt.plot(t_data, wxh_evalx, label="wxh_x",color='r')
# plt.xlabel("t")
# plt.ylabel("wxh_x")
# plt.grid(True)
# plt.legend()

# plt.subplot(3, 1, 2)
# plt.plot(t_data, wxh_evaly, label="wxh_y",color='g')
# plt.xlabel("t")
# plt.ylabel(" wxh_y")
# plt.grid(True)
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(t_data, wxh_evalz, label="wxh_z",color='b')
# plt.xlabel("t")
# plt.ylabel("wxh_z")
# plt.grid(True)
# plt.legend()

# plt.tight_layout()



plt.show()

# ---------------------------------------------------------------------------------------------------------------- #


print('Codice terminato')
