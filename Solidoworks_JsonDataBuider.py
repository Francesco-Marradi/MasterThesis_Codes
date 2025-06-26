#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI-----------------------------------------------#
#------------------------------------------- SCRIPT OBJECTIVE ---------------------------------------------------------#

# The objective of this script is to derive the eq. of motion of a system by applying kane's algorithm.

# The script will be adjusted case by case by the user, following the instruction prompted on the terminal

# Kane's algorithm stars by defining the position vectors of the center of mass of the bodies, and then
# derive them along time to obtain the velocity and acceleration vectors for the two points.
# We then derive the partial velocity vectors from the total velocity vectors of the two center of mass, and then
# we proceed to calculate the generalized forces acting on our system.
# We then proceed to assemble's kane's eq. of motion to obtain, by symplification, the eq of motion of the system.

# This is going to be version 1 of "Kane's method matrix form rearranged"


#------------------------------------ IMPORTING THE NECESSARY PACKAGES -------------------------------------------------#

import os
import csv
import json
import numpy as np


# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

#Cleaning the terminal 
os.system('cls')



# -------------------------------------------- LOADING DATA FROM THE SOLVER OUTPUT ----------------------------------------------- #

# Caricamento dei dati dal file JSON
with open('solution_4arms_Preloadedsprings_dampers_FINAL.json', 'r') as f:
    data_loaded = json.load(f)

# Estrazione dei dati dal JSON
t_loaded = np.array(data_loaded["t"]) 
y_loaded = np.array(data_loaded["y"])

'''
# 1 ARM FIXED BUS CODE
# Unpacking the datas from the vector y_loaded
q4_sol = y_loaded[4,:] 
q7_sol = y_loaded[5,:]
q10_sol = y_loaded[6,:]
q13_sol = y_loaded[7,:]
'''

# 4 ARMS CODE
# Unpacking the datas from the vector y_loaded
u1_sol = y_loaded[0,:]
u2_sol = y_loaded[1,:]
u3_sol = y_loaded[2,:]

q1_sol = y_loaded[22,:]
q2_sol = y_loaded[23,:]
q3_sol = y_loaded[24,:]
q4_sol = y_loaded[25,:]
q7_sol = y_loaded[26,:]
q10_sol = y_loaded[27,:]
q13_sol = y_loaded[28,:]
q18_sol = y_loaded[29,:]
q21_sol = y_loaded[30,:]
q24_sol = y_loaded[31,:]
q27_sol = y_loaded[32,:]
q28_sol = y_loaded[33,:]
q31_sol = y_loaded[34,:]
q34_sol= y_loaded[35,:]
q37_sol = y_loaded[36,:]
q42_sol = y_loaded[37,:]
q45_sol = y_loaded[38,:]
q48_sol = y_loaded[39,:]
q51_sol = y_loaded[40,:]
q52_sol = y_loaded[41,:]
q53_sol = y_loaded[42,:]
q54_sol = y_loaded[43,:]



# --------------------------------------------- DATA SAMPLING FOR SOLIDWORKS ANIMATION ------------------------------------------------ #

# Here we want to sample the datas from the solver output. In this way we can animate the solidworks cad by manipolation of the bodies 
# positions.

# Sampling time
t_sample = 0.05      # [s]

# Sampling interval calculations
Solver_tot_time = t_loaded[-1]
Sampling_interval = t_sample/(Solver_tot_time/len(t_loaded))

# Solver solutions Sampling
t_sampled = t_loaded[::int(round(Sampling_interval))]

# Spacecraft Bus
# Angular displacement
q1_sampled = q1_sol[::int(round(Sampling_interval))]
q2_sampled = q2_sol[::int(round(Sampling_interval))]
q3_sampled = q3_sol[::int(round(Sampling_interval))]

# Angular velocities
u1_sampled = u1_sol[::int(round(Sampling_interval))]
u2_sampled = u2_sol[::int(round(Sampling_interval))]
u3_sampled = u3_sol[::int(round(Sampling_interval))]

# Linear displacement
q52_sampled = q52_sol[::int(round(Sampling_interval))]
q53_sampled = q53_sol[::int(round(Sampling_interval))]
q54_sampled = q54_sol[::int(round(Sampling_interval))]


# Robotic Arm 1
q4_sampled = q4_sol[::int(round(Sampling_interval))]
q7_sampled = q7_sol[::int(round(Sampling_interval))]
q10_sampled = q10_sol[::int(round(Sampling_interval))]
q13_sampled = q13_sol[::int(round(Sampling_interval))]

# Robotic Arm 2
q18_sampled = q18_sol[::int(round(Sampling_interval))]
q21_sampled = q21_sol[::int(round(Sampling_interval))]
q24_sampled = q24_sol[::int(round(Sampling_interval))]
q27_sampled = q27_sol[::int(round(Sampling_interval))]

# Robotic Arm 3
q28_sampled = q28_sol[::int(round(Sampling_interval))]
q31_sampled = q31_sol[::int(round(Sampling_interval))]
q34_sampled = q34_sol[::int(round(Sampling_interval))]
q37_sampled = q37_sol[::int(round(Sampling_interval))]

# Robotic Arm 4
q42_sampled = q42_sol[::int(round(Sampling_interval))]
q45_sampled = q45_sol[::int(round(Sampling_interval))]
q48_sampled = q48_sol[::int(round(Sampling_interval))]
q51_sampled = q51_sol[::int(round(Sampling_interval))]


# Since the angles in solidworks are absolute and not relative, i need to add at each leaf body the angle deployment of the precedent one
# Robotic Arm 1
q4_sampled_abs = q4_sampled + q3_sampled 
q7_sampled_abs = q7_sampled + q4_sampled + q3_sampled 
q10_sampled_abs = q10_sampled + q7_sampled + q4_sampled + q3_sampled 
q13_sampled_abs = q13_sampled + q10_sampled + q7_sampled + q4_sampled + q3_sampled 

# Robotic Arm 2
q18_sampled_abs = q18_sampled + q1_sampled
q21_sampled_abs = q21_sampled + q18_sampled + q1_sampled
q24_sampled_abs = q24_sampled + q21_sampled + q18_sampled + q1_sampled
q27_sampled_abs = q27_sampled + q24_sampled + q21_sampled + q18_sampled + q1_sampled

# Robotic Arm 3
q28_sampled_abs = q28_sampled + q3_sampled 
q31_sampled_abs = q31_sampled + q28_sampled + q3_sampled 
q34_sampled_abs = q34_sampled + q31_sampled + q28_sampled + q3_sampled 
q37_sampled_abs = q37_sampled + q34_sampled + q31_sampled + q28_sampled + q3_sampled 

# Robotic Arm 4
q42_sampled_abs = q42_sampled + q1_sampled
q45_sampled_abs = q45_sampled + q42_sampled + q1_sampled
q48_sampled_abs = q48_sampled + q45_sampled + q42_sampled + q1_sampled
q51_sampled_abs = q51_sampled + q48_sampled + q45_sampled + q42_sampled + q1_sampled



# ---------------------------------------------- WRITING THE CSV FILES -------------------------------------------------- # 
# At last, we need to produce a .csv file for each motors, loadable by solidworks with all the information

# Spacecraft bus
# Rotation
with open('rotary_motorB0x.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q1_sampled*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB0y.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q2_sampled*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB0z.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q3_sampled*(180/np.pi)):
        writer.writerow([t, angle])


with open('rotary_motorB0x_dot.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle_s in zip(t_sampled, u1_sampled*(180/np.pi)):
        writer.writerow([t, angle_s])

with open('rotary_motorB0y_dot.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle_s in zip(t_sampled, u2_sampled*(180/np.pi)):
        writer.writerow([t, angle_s])

with open('rotary_motorB0z_dot.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle_s in zip(t_sampled, u3_sampled*(180/np.pi)):
        writer.writerow([t, angle_s])


# Traslation
with open('linear_motorB0x.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q52_sampled*(10**(3))):
        writer.writerow([t, angle])

with open('linear_motorB0y.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q53_sampled*(10**(3))):
        writer.writerow([t, angle])

with open('linear_motorB0z.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q54_sampled*(10**(3))):
        writer.writerow([t, angle])



# Robotic arm 1
with open('rotary_motorB11.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q4_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB21.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q7_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB31.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q10_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB41.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q13_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])


# Robotic arm 2
with open('rotary_motorB12.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q18_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB22.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q21_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB32.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q24_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB42.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q27_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])


# Robotic arm 3
with open('rotary_motorB13.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q28_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB23.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q31_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB33.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q34_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB43.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q37_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])


# Robotic arm 4
with open('rotary_motorB14.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q42_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB24.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q45_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB34.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q48_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])

with open('rotary_motorB44.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t, angle in zip(t_sampled, q51_sampled_abs*(180/np.pi)):
        writer.writerow([t, angle])


print('Codice terminato')