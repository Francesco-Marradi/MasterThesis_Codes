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

import sympy as sy
import sympy.physics.mechanics as me
import os
import itertools
from scipy.io import savemat
import json
import re
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

#Cleaning the terminal 
os.system('cls')


print('ciao977')

#---------------------------------------- SYSTEM GENERAL PARAMETERS -------------------------------------------------#   

nb = 17                           # Number of bodies present 
nbranch = 4                       # Number of branches
nb_branch = 5                     # Number of bodies in each branch
njoint_in_branch = 4              # Number of joints in each branch
nj = nbranch*njoint_in_branch     # Number of joints presents


# Calculations
n_g = 6 + 3*(nj)        # Numer of gen. coordinates and speeds to define


# For the sake of brevity, we define "Ii" as the identity 3x3 matrix
Ii = sy.eye(3)

# For the sake of even more brevity, we define "II_vec" as a vertical stack of identity 3x3 matrix, forming a 12x3 matrix
II_vec = sy.Matrix.vstack(Ii,Ii,Ii,Ii)

# For the sake of brevity, we define "Io" as a zeros 3x3 matrix
Io = sy.zeros(3,3)

# For the sake of more brevity, we define "IO" as a zeros 12x12 matrix
IO = sy.zeros(12,12)

# For the sake of even more brevity, we define "IO_vec" as a zeros 12x3 matrix
IO_vec = sy.zeros(12,3)


#------------------------------------------ SYMBOLIC VARIABLES ----------------------------------------------------#    

# Initial motion parameters
t = sy.symbols('t')                 # Definition of the time variable
l = sy.symbols('l')                 # Spacecraft bus length [m]
w = sy.symbols('w')                 # Spacecraft bus width [m]
h = sy.symbols('h')                 # Spacecraft bus height [m]
d = sy.symbols('d')                 # Spacecraft bus-arm joint distance [m]
dg = sy.symbols('dg')               # Spacecraft bus joint distance [m]
bl = sy.symbols('bl')               # Robotic arm long piece length [m]
hl = sy.symbols('hl')               # Robotic arm long piece height [m]
wl = sy.symbols('wl')               # Robotic arm long piece width [m]
bs = sy.symbols('bs')               # Robotic arm short piece length [m]
hs = sy.symbols('hs')               # Robotic arm short piece height [m]
ws = sy.symbols('ws')               # Robotic arm short piece width [m]
mlongarm = sy.symbols('mlongarm')   # Robotic Long arm piece mass [kg]
mshortarm = sy.symbols('mshortarm') # Robotic Short arm piece mass [kg]
mbus = sy.symbols('mbus')           # Spacecraft bus mass [kg]


# Forces and Torques applied
T11 = sy.symbols('T11')             # Torque acting on the arms [N*m]
T21 = sy.symbols('T21')             # Torque acting on the arms [N*m]
T31 = sy.symbols('T31')             # Torque acting on the arms [N*m]
T41 = sy.symbols('T41')             # Torque acting on the arms [N*m]
T12 = sy.symbols('T12')             # Torque acting on the arms [N*m]
T22 = sy.symbols('T22')             # Torque acting on the arms [N*m]
T32 = sy.symbols('T32')             # Torque acting on the arms [N*m]
T42 = sy.symbols('T42')             # Torque acting on the arms [N*m]
T13 = sy.symbols('T13')             # Torque acting on the arms [N*m]
T23 = sy.symbols('T23')             # Torque acting on the arms [N*m]
T33 = sy.symbols('T33')             # Torque acting on the arms [N*m]
T43 = sy.symbols('T43')             # Torque acting on the arms [N*m]
T14 = sy.symbols('T14')             # Torque acting on the arms [N*m]
T24 = sy.symbols('T24')             # Torque acting on the arms [N*m]
T34 = sy.symbols('T34')             # Torque acting on the arms [N*m]
T44 = sy.symbols('T44')             # Torque acting on the arms [N*m]
F = sy.symbols('F')                 # Force acting on the arms [N*m]


# Forces and Torques constants
k11 = sy.symbols('k11')             # B11 body spring's stiffness [N*m/rad]
k21 = sy.symbols('k21')             # B21 body spring's stiffness [N*m/rad] 
k31 = sy.symbols('k31')             # B31 body spring's stiffness [N*m/rad] 
k41 = sy.symbols('k41')             # B41 body spring's stiffness [N*m/rad]
k12 = sy.symbols('k11')             # B12 body spring's stiffness [N*m/rad]
k22 = sy.symbols('k21')             # B22 body spring's stiffness [N*m/rad] 
k32 = sy.symbols('k31')             # B32 body spring's stiffness [N*m/rad] 
k42 = sy.symbols('k41')             # B42 body spring's stiffness [N*m/rad] 
k13 = sy.symbols('k11')             # B13 body spring's stiffness [N*m/rad]
k23 = sy.symbols('k21')             # B23 body spring's stiffness [N*m/rad] 
k33 = sy.symbols('k31')             # B33 body spring's stiffness [N*m/rad] 
k43 = sy.symbols('k41')             # B43 body spring's stiffness [N*m/rad] 
k14 = sy.symbols('k11')             # B14 body spring's stiffness [N*m/rad]
k24 = sy.symbols('k21')             # B24 body spring's stiffness [N*m/rad] 
k34 = sy.symbols('k31')             # B34 body spring's stiffness [N*m/rad] 
k44 = sy.symbols('k41')             # B44 body spring's stiffness [N*m/rad] 

c11 = sy.symbols('c11')             # B11 body damping coefficient [(N*m/s)/(rad/s)]
c21 = sy.symbols('c21')             # B21 body damping coefficient [(N*m/s)/(rad/s)]
c31 = sy.symbols('c31')             # B31 body damping coefficient [(N*m/s)/(rad/s)]
c41 = sy.symbols('c41')             # B41 body damping coefficient [(N*m/s)/(rad/s)]
c12 = sy.symbols('c11')             # B12 body damping coefficient [(N*m/s)/(rad/s)]
c22 = sy.symbols('c21')             # B22 body damping coefficient [(N*m/s)/(rad/s)]
c32 = sy.symbols('c31')             # B32 body damping coefficient [(N*m/s)/(rad/s)]
c42 = sy.symbols('c41')             # B42 body damping coefficient [(N*m/s)/(rad/s)]
c13 = sy.symbols('c11')             # B13 body damping coefficient [(N*m/s)/(rad/s)]
c23 = sy.symbols('c21')             # B23 body damping coefficient [(N*m/s)/(rad/s)]
c33 = sy.symbols('c31')             # B33 body damping coefficient [(N*m/s)/(rad/s)]
c43 = sy.symbols('c41')             # B43 body damping coefficient [(N*m/s)/(rad/s)]
c14 = sy.symbols('c11')             # B14 body damping coefficient [(N*m/s)/(rad/s)]
c24 = sy.symbols('c21')             # B24 body damping coefficient [(N*m/s)/(rad/s)]
c34 = sy.symbols('c31')             # B34 body damping coefficient [(N*m/s)/(rad/s)]
c44 = sy.symbols('c41')             # B44 body damping coefficient [(N*m/s)/(rad/s)]



# Deployment spring dampers
# Deployment angles 
# Robotic arm 1
q4_deploy = (-30)*(np.pi/180)
q7_deploy = (-120)*(np.pi/180)
q10_deploy = (-60)*(np.pi/180)
q13_deploy = (-15)*(np.pi/180)

# Robotic arm 2
q18_deploy = (30)*(np.pi/180)
q21_deploy = (120)*(np.pi/180)
q24_deploy = (60)*(np.pi/180)
q27_deploy = (15)*(np.pi/180)

# Robotic arm 3
q28_deploy = (30)*(np.pi/180)
q31_deploy = (120)*(np.pi/180)
q34_deploy = (60)*(np.pi/180)
q37_deploy = (15)*(np.pi/180)

# Robotic arm 4
q42_deploy = (-30)*(np.pi/180)
q45_deploy = (-120)*(np.pi/180)
q48_deploy = (-60)*(np.pi/180)
q51_deploy = (-15)*(np.pi/180)



# ---------------------------------- GENERALIZED COORDINATES AND SPEEDS DEFINITION ------------------------------------- #

# Generalized Coordinates and Speeds definition. These are going to be putted inside two lists, creating the Generalized System
# body state vector and State Vector
Gen_coord_list = [me.dynamicsymbols(f'q{i+1}') for i in range(n_g)]
Gen_speeds_list = [me.dynamicsymbols(f'u{i+1}') for i in range(n_g)]


# We need also to define them outside of a list, so they can be  used to define ref. systems and forces
for i in range(n_g):
    exec(f'q{i+1} = me.dynamicsymbols(f"q{i+1}")')
    exec(f'u{i+1} = me.dynamicsymbols(f"u{i+1}")')



# Dictionary to subsitute the time derivatives of the gen. coordinates with the gen. speeds
gencoordDer_to_speeds_dict = dict(zip(sy.Matrix([Gen_coord_list[3:]]).diff(), sy.Matrix([Gen_speeds_list[3:]]))) 

# Dictionary to subsitute the double time derivatives of the gen. coordinates with the time derivative of the gen. speeds
gencoordDerDer_to_speedsDer_dict = dict(zip(sy.Matrix([Gen_coord_list[3:]]).diff(t,2), sy.Matrix([Gen_speeds_list[3:]]).diff(t)))



#----------------------------------------- REFERENCE FRAME DEFINITION --------------------------------------------------#
print('Defining reference frames positions and orientations...')


# Inertial Reference Frame definition
n = me.ReferenceFrame('n')          # Inertial ref. Frame, fixed with   

# Bodies Ref. Frame definitions
b00 = me.ReferenceFrame('b00')      # Spacecraft bus ref. Frame, with origin centered on point B0 [root body], used for inertia definitions
b0 = me.ReferenceFrame('b0')        # Spacecraft bus ref. Frame, with origin centered on point B0 [root body] and rotated of 45°



# -------------------------------------------------------------- #
# Robotic arms ref. frames definitions

# Robotic arm 1
b11 = me.ReferenceFrame('b11')        # Robotic arm long piece ref. Frame, with origin centered on point B11
b21 = me.ReferenceFrame('b21')        # Robotic arm long piece ref. Frame, with origin centered on point B21
b31 = me.ReferenceFrame('b31')        # Robotic arm long piece ref. Frame, with origin centered on point B31
b41 = me.ReferenceFrame('b41')        # Robotic arm short piece ref. Frame, with origin centered on point B41

# Robotic arm 2
b12 = me.ReferenceFrame('b12')        # Robotic arm long piece ref. Frame, with origin centered on point B12
b22 = me.ReferenceFrame('b22')        # Robotic arm long piece ref. Frame, with origin centered on point B22
b32 = me.ReferenceFrame('b32')        # Robotic arm long piece ref. Frame, with origin centered on point B32
b42 = me.ReferenceFrame('b42')        # Robotic arm short piece ref. Frame, with origin centered on point B42

# Robotic arm 3
b13 = me.ReferenceFrame('b13')        # Robotic arm long piece ref. Frame, with origin centered on point B13
b23 = me.ReferenceFrame('b23')        # Robotic arm long piece ref. Frame, with origin centered on point B23
b33 = me.ReferenceFrame('b33')        # Robotic arm long piece ref. Frame, with origin centered on point B33
b43 = me.ReferenceFrame('b43')        # Robotic arm short piece ref. Frame, with origin centered on point B43

# Robotic arm 4
b14 = me.ReferenceFrame('b14')        # Robotic arm long piece ref. Frame, with origin centered on point B14
b24 = me.ReferenceFrame('b24')        # Robotic arm long piece ref. Frame, with origin centered on point B24
b34 = me.ReferenceFrame('b34')        # Robotic arm long piece ref. Frame, with origin centered on point B34
b44 = me.ReferenceFrame('b44')        # Robotic arm short piece ref. Frame, with origin centered on point B44


# -------------------------------------------------------------- #
# Inertial Reference Frame origin definition
O = me.Point('O')                   # Inertial ref. Frame origin

# Bodies Ref. Frame origin definition
B0 = me.Point('B0')                 # Spacecraft bus ref. Frame origin


# -------------------------------------------------------------- #
# Robotic arms ref. frame origins definition

# Robotic arm 1
B11 = me.Point('B11')                 # Robotic arm long piece ref. Frame origin
B21 = me.Point('B21')                 # Robotic arm long piece ref. Frame origin 
B31 = me.Point('B31')                 # Robotic arm long piece ref. Frame origin 
B41 = me.Point('B41')                 # Robotic arm long piece ref. Frame origin 

# Robotic arm 2
B12 = me.Point('B12')                 # Robotic arm long piece ref. Frame origin
B22 = me.Point('B22')                 # Robotic arm long piece ref. Frame origin 
B32 = me.Point('B32')                 # Robotic arm long piece ref. Frame origin 
B42 = me.Point('B42')                 # Robotic arm long piece ref. Frame origin 

# Robotic arm 3
B13 = me.Point('B13')                 # Robotic arm long piece ref. Frame origin
B23 = me.Point('B23')                 # Robotic arm long piece ref. Frame origin 
B33 = me.Point('B33')                 # Robotic arm long piece ref. Frame origin 
B43 = me.Point('B43')                 # Robotic arm long piece ref. Frame origin 

# Robotic arm 4
B14 = me.Point('B14')                 # Robotic arm long piece ref. Frame origin
B24 = me.Point('B24')                 # Robotic arm long piece ref. Frame origin 
B34 = me.Point('B34')                 # Robotic arm long piece ref. Frame origin 
B44 = me.Point('B44')                 # Robotic arm long piece ref. Frame origin 


# -------------------------------------------------------------- #
# Joints points definitions

# Robotic arm 1
G01 = me.Point('G01')                 # Joint G01
G11 = me.Point('G11')                 # Joint G11
G21 = me.Point('G21')                 # Joint G21
G31 = me.Point('G31')                 # Joint G31

# Robotic arm 2
G02 = me.Point('G02')                 # Joint G02
G12 = me.Point('G12')                 # Joint G12
G22 = me.Point('G22')                 # Joint G22
G32 = me.Point('G32')                 # Joint G32

# Robotic arm 3
G03 = me.Point('G03')                 # Joint G03
G13 = me.Point('G13')                 # Joint G13
G23 = me.Point('G23')                 # Joint G23
G33 = me.Point('G33')                 # Joint G33

# Robotic arm 4
G04 = me.Point('G04')                 # Joint G04
G14 = me.Point('G14')                 # Joint G14
G24 = me.Point('G24')                 # Joint G24
G34 = me.Point('G34')                 # Joint G34


# -------------------------------------------------------------- #
# Setting the relative position between the bodies and joints

# Spacecraft bus
B0.set_pos(O, q52*n.x + q53*n.y + q54*n.z)                    # Setting point B0 relative to the inertial ref. Frame
B0.set_pos(B0, 0*b0.x + 0*b0.y + 0*b0.z)                      # Setting point B0 as the origin of the root rotated ref. frame
B0.set_pos(B0, 0*b00.x + 0*b00.y + 0*b00.z)                   # Setting point B0 as the origin of the root ref. frame

# Robotic arm 1
G01.set_pos(B0, (l/2 + dg)*b0.y + d*b0.z)                     # Setting the joint G01 position wrt spacecraft bus center of mass
B11.set_pos(G01, (bl/2)*b11.z)                                # Setting the body B11 position wrt the joint G01
G11.set_pos(B11, (bl/2)*b11.z)                                # Setting the joint G11 position wrt the body B11
B21.set_pos(G11, -(bl/2)*b21.y)                               # Setting the body B21 position wrt the joint G11
G21.set_pos(B21, -(bl/2)*b21.y)                               # Setting the joint G21 position wrt the body B21
B31.set_pos(G21, -(bl/2)*b31.y)                               # Setting the body B31 position wrt the joint G21
G31.set_pos(B31, -(bl/2)*b31.y)                               # Setting the joint G31 position wrt the body B31
B41.set_pos(G31, -(bs/2)*b41.y)                               # Setting the body B31 position wrt the joint G21

# Robotic arm 2
G02.set_pos(B0, (l/2 + dg)*b0.y + d*b0.x)                     # Setting the joint G02 position wrt spacecraft bus center of mass
B12.set_pos(G02, (bl/2)*b12.x)                                # Setting the body B12 position wrt the joint G02
G12.set_pos(B12, (bl/2)*b12.x)                                # Setting the joint G12 position wrt the body B12
B22.set_pos(G12, -(bl/2)*b22.y)                               # Setting the body B22 position wrt the joint G12
G22.set_pos(B22, -(bl/2)*b22.y)                               # Setting the joint G22 position wrt the body B22
B32.set_pos(G22, -(bl/2)*b32.y)                               # Setting the body B32 position wrt the joint G22
G32.set_pos(B32, -(bl/2)*b32.y)                               # Setting the joint G32 position wrt the body B32
B42.set_pos(G32, -(bs/2)*b42.y)                               # Setting the body B32 position wrt the joint G22

# Robotic arm 3
G03.set_pos(B0, (l/2 + dg)*b0.y - d*b0.z)                     # Setting the joint G03 position wrt spacecraft bus center of mass
B13.set_pos(G03, -(bl/2)*b13.z)                               # Setting the body B13 position wrt the joint G03
G13.set_pos(B13, -(bl/2)*b13.z)                               # Setting the joint G13 position wrt the body B13
B23.set_pos(G13, -(bl/2)*b23.y)                               # Setting the body B23 position wrt the joint G13
G23.set_pos(B23, -(bl/2)*b23.y)                               # Setting the joint G23 position wrt the body B23
B33.set_pos(G23, -(bl/2)*b33.y)                               # Setting the body B33 position wrt the joint G23
G33.set_pos(B33, -(bl/2)*b33.y)                               # Setting the joint G33 position wrt the body B33
B43.set_pos(G33, -(bs/2)*b43.y)                               # Setting the body B33 position wrt the joint G23

# Robotic arm 4
G04.set_pos(B0, (l/2 + dg)*b0.y - d*b0.x)                     # Setting the joint G04 position wrt spacecraft bus center of mass
B14.set_pos(G04, -(bl/2)*b14.x)                               # Setting the body B14 position wrt the joint G04
G14.set_pos(B14, -(bl/2)*b14.x)                               # Setting the joint G14 position wrt the body B14
B24.set_pos(G14, -(bl/2)*b24.y)                               # Setting the body B24 position wrt the joint G14
G24.set_pos(B24, -(bl/2)*b24.y)                               # Setting the joint G24 position wrt the body B24
B34.set_pos(G24, -(bl/2)*b34.y)                               # Setting the body B34 position wrt the joint G24
G34.set_pos(B34, -(bl/2)*b34.y)                               # Setting the joint G34 position wrt the body B34
B44.set_pos(G34, -(bs/2)*b44.y)                               # Setting the body B34 position wrt the joint G24


# Setting each body cog as the center of its own body ref.frame
# Robotic arm 1
B11.set_pos(B11, 0*b11.x + 0*b11.y + 0*b11.z) 
B21.set_pos(B21, 0*b21.x + 0*b21.y + 0*b21.z)
B31.set_pos(B31, 0*b31.x + 0*b31.y + 0*b31.z)
B41.set_pos(B41, 0*b41.x + 0*b41.y + 0*b41.z)

# Robotic arm 2
B12.set_pos(B12, 0*b12.x + 0*b12.y + 0*b12.z) 
B22.set_pos(B22, 0*b22.x + 0*b22.y + 0*b22.z)
B32.set_pos(B32, 0*b32.x + 0*b32.y + 0*b32.z)
B42.set_pos(B42, 0*b42.x + 0*b42.y + 0*b42.z)

# Robotic arm 3
B13.set_pos(B13, 0*b13.x + 0*b13.y + 0*b13.z) 
B23.set_pos(B23, 0*b23.x + 0*b23.y + 0*b23.z)
B33.set_pos(B33, 0*b33.x + 0*b33.y + 0*b33.z)
B43.set_pos(B43, 0*b43.x + 0*b43.y + 0*b43.z)

# Robotic arm 4
B14.set_pos(B14, 0*b14.x + 0*b14.y + 0*b14.z) 
B24.set_pos(B24, 0*b24.x + 0*b24.y + 0*b24.z)
B34.set_pos(B34, 0*b34.x + 0*b34.y + 0*b34.z)
B44.set_pos(B44, 0*b44.x + 0*b44.y + 0*b44.z)

# -------------------------------------------------------------- #
# Setting the relative orientantion between the bodies reference frames

# Defining the spacecraft bus non-rotated ref. frame
b00.orient_body_fixed(n, (q1, q2, q3), 'zxy')

# Defining the spacecraft bus ref. frame as rotated wrt b0
b0.orient_body_fixed(b00, (0, 0, (- sy.pi/4)), 'zxy')

# Robotic arm 1
b11.orient_body_fixed(b0, (q6, q4, q5), 'zxy')                  
b21.orient_body_fixed(b11, (q9, q7, q8), 'zxy')
b31.orient_body_fixed(b21, (q12, q10, q11), 'zxy')
b41.orient_body_fixed(b31, (q15, q13, q14), 'zxy')

# Robotic arm 2
b12.orient_body_fixed(b0, (q18, q16, q17), 'zxy')                  
b22.orient_body_fixed(b12, (q21, q19, q20), 'zxy')
b32.orient_body_fixed(b22, (q24, q22, q23), 'zxy')
b42.orient_body_fixed(b32, (q27, q25, q26), 'zxy')

# Robotic arm 3
b13.orient_body_fixed(b0, (q30, q28, q29), 'zxy')                  
b23.orient_body_fixed(b13, (q33, q31, q32), 'zxy')
b33.orient_body_fixed(b23, (q36, q34, q35), 'zxy')
b43.orient_body_fixed(b33, (q39, q37, q38), 'zxy')

# Robotic arm 4
b14.orient_body_fixed(b0, (q42, q40, q41), 'zxy')                  
b24.orient_body_fixed(b14, (q45, q43, q44), 'zxy')
b34.orient_body_fixed(b24, (q48, q46, q47), 'zxy')
b44.orient_body_fixed(b34, (q51, q49, q50), 'zxy')


# -------------------------------------------------------------- #
# Setting the velocities of the root body
# B0.set_vel(n, u52*n.x + u53*n.y + u54*n.z)
b00.set_ang_vel(n, u1*b00.x + u2*b00.y + u3*b00.z)


# Setting the velocity of the inertial ref. frame to 0, to allow the calculations to be developed
O.set_vel(n, 0)
n.set_ang_vel(n, 0)



# ---------------------------------------- SUBSTITUTION DICTIONARIES ------------------------------------------------ #
# Dictionary
substitution = {
    l: 0.6,                 # Spacecraft bus length [m]
    w: 0.5,                 # Spacecraft bus width [m]
    h: 0.5,                 # Spacecraft bus height [m]
    d: 0.1,                 # Spacecraft bus-arm joint distance [m]
    dg: 0.025,              # Spacecraft bus joint distance [m]
    bl: 0.25,               # Robotic arm long piece length [m]
    hl: 0.05,               # Robotic arm long piece height [m]
    wl: 0.05,               # Robotic arm long piece width [m]
    bs: 0.16,               # Robotic arm short piece length [m]
    hs: 0.05,               # Robotic arm short piece height [m]
    ws: 0.05,               # Robotic arm short piece width [m]
    F: 0.0,                 # Force acting on the arms [N]
    mlongarm: 0.510,        # Robotic Long arm piece mass [kg]
    mshortarm: 0.320,       # Robotic Short arm piece mass [kg]
    mbus: 121.2,            # Spacecraft bus mass [kg]

    # Setting to zero the torques
    T11 : 0,
    T21 : 0,
    T31 : 0,
    T41 : 0,
    T12 : 0,
    T22 : 0,
    T32 : 0,
    T42 : 0,
    T13 : 0,
    T23 : 0,
    T33 : 0,
    T43 : 0,
    T14 : 0,
    T24 : 0,
    T34 : 0,
    T44 : 0,

}



# ------------------------------------------ LOADING DATA FOR ANIMATION -------------------------------------------------- #
# Caricamento dei dati dal file JSON
with open('solution_4arms_Preloadedsprings_dampers_RotatingSc.json', 'r') as f:
    data_loaded = json.load(f)

# Loading the baricenter position wrt B0
with open('Satellite_baricenter_4arms_Preloadedsprings_dampers_vect.json', 'r') as f:
    data_baricenter = json.load(f)


# Estrazione dei dati dal JSON
t_loaded = np.array(data_loaded["t"])  # Converti di nuovo in numpy array
y_loaded = np.array(data_loaded["y"])
baricenter_loaded = sy.sympify(data_baricenter['Satellite_baricenter']['Satellite_baricenter'], locals = {'n':n,'b00':b00,'b0':b0,'b11':b11,'b21':b21,'b31':b31,'b41':b41,'b12':b12,'b22':b22,'b32':b32,'b42':b42,'b13':b13,'b23':b23,'b33':b33,'b43':b43,'b14':b14,'b24':b24,'b34':b34,'b44':b44})



# Unpacking the datas from the vector y_loaded
u1_sol = y_loaded[0,:]
u2_sol = y_loaded[1,:]
u3_sol = y_loaded[2,:]
u4_sol = y_loaded[3,:]
u7_sol = y_loaded[4,:]
u10_sol = y_loaded[5,:]
u13_sol = y_loaded[6,:]
u18_sol = y_loaded[7,:]
u21_sol = y_loaded[8,:]
u24_sol = y_loaded[9,:]
u27_sol = y_loaded[10,:]
u28_sol = y_loaded[11,:]
u31_sol = y_loaded[12,:]
u34_sol = y_loaded[13,:]
u37_sol = y_loaded[14,:]
u42_sol = y_loaded[15,:]
u45_sol = y_loaded[16,:]
u48_sol = y_loaded[17,:]
u51_sol = y_loaded[18,:]
u52_sol = y_loaded[19,:]
u53_sol = y_loaded[20,:]
u54_sol = y_loaded[21,:]


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
q34_sol = y_loaded[35,:]
q37_sol = y_loaded[36,:] 
q42_sol = y_loaded[37,:]
q45_sol = y_loaded[38,:]
q48_sol = y_loaded[39,:] 
q51_sol = y_loaded[40,:]
q52_sol = y_loaded[41,:]
q53_sol = y_loaded[42,:]
q54_sol = y_loaded[43,:]



# Defining the COG as a point
COG = me.Point('COG')

# Setting the position of the center of mass wrt the inertial ref frame
COG.set_pos(O, baricenter_loaded)



# ------------------------------------------------------ PLOTTING ---------------------------------------------------------------- #
# Create a list to store the positions
positions = []
colors = []  # Create a list to store colors

# Create figure for plotting
fig = plt.figure(figsize=(10, 8)) 
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
ax = fig.add_subplot(111, projection='3d')
time_text = fig.text(0.1, 0.9, '')


# --------------------------------------------------------- ANIMATION ----------------------------------------------------------- #

# Turning off unnecessary dof
# Calcola le nuove posizioni dei punti in base al frame
gencoord_active = sy.Matrix([q1, q2, q3, q4, 0, 0, q7, 0, 0, q10, 0, 0, q13, 0, 0, 0, 0, q18, 0, 0, q21, 0, 0, q24, 0, 0, q27, q28, 0, 0, q31, 0, 0, q34, 0, 0, q37, 0, 0, 0, 0, q42, 0, 0, q45, 0, 0, q48, 0, 0, q51, q52, q53, q54])

# Constructing the dictionaries
gencoord_active_dict = dict(zip(Gen_coord_list ,gencoord_active))

# To optimize the animation, first we calculate the symbolic position of each point, then we use lambdify to convert it into a numpy array,
# and at last, we evaluate it

allvar = (q1, q2, q3, q4, q7, q10, q13, q18, q21, q24, q27, q28, q31, q34, q37, q42, q45, q48, q51, q52, q53, q54) 
B0_lamb_vect = sy.lambdify(allvar,B0.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
COG_lambd_vect = sy.lambdify(allvar,COG.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')

# Robotic arm 1
G01_lamb_vect = sy.lambdify(allvar,G01.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B11_lamb_vect = sy.lambdify(allvar,B11.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G11_lamb_vect = sy.lambdify(allvar,G11.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B21_lamb_vect = sy.lambdify(allvar,B21.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G21_lamb_vect = sy.lambdify(allvar,G21.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B31_lamb_vect = sy.lambdify(allvar,B31.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G31_lamb_vect = sy.lambdify(allvar,G31.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B41_lamb_vect = sy.lambdify(allvar,B41.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')

# Robotic arm 2
G02_lamb_vect = sy.lambdify(allvar,G02.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B12_lamb_vect = sy.lambdify(allvar,B12.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G12_lamb_vect = sy.lambdify(allvar,G12.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B22_lamb_vect = sy.lambdify(allvar,B22.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G22_lamb_vect = sy.lambdify(allvar,G22.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B32_lamb_vect = sy.lambdify(allvar,B32.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G32_lamb_vect = sy.lambdify(allvar,G32.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B42_lamb_vect = sy.lambdify(allvar,B42.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')

# Robotic arm 3
G03_lamb_vect = sy.lambdify(allvar,G03.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B13_lamb_vect = sy.lambdify(allvar,B13.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G13_lamb_vect = sy.lambdify(allvar,G13.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B23_lamb_vect = sy.lambdify(allvar,B23.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G23_lamb_vect = sy.lambdify(allvar,G23.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B33_lamb_vect = sy.lambdify(allvar,B33.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G33_lamb_vect = sy.lambdify(allvar,G33.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B43_lamb_vect = sy.lambdify(allvar,B43.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')

# Robotic arm 4
G04_lamb_vect = sy.lambdify(allvar,G04.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B14_lamb_vect = sy.lambdify(allvar,B14.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G14_lamb_vect = sy.lambdify(allvar,G14.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B24_lamb_vect = sy.lambdify(allvar,B24.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G24_lamb_vect = sy.lambdify(allvar,G24.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B34_lamb_vect = sy.lambdify(allvar,B34.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
G34_lamb_vect = sy.lambdify(allvar,G34.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')
B44_lamb_vect = sy.lambdify(allvar,B44.pos_from(O).to_matrix(n).xreplace(substitution).xreplace(gencoord_active_dict), 'numpy')



# # ------------------------------------------------- REFERENCE FRAMES ------------------------------------------------------------ #
# # Inertial ref frame
# x_n_lamb = sy.lambdify(allvar,O.pos_from(O).to_matrix(n)[0].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# y_n_lamb = sy.lambdify(allvar,O.pos_from(O).to_matrix(n)[1].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# z_n_lamb = sy.lambdify(allvar,O.pos_from(O).to_matrix(n)[2].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# xv_n_lamb = sy.lambdify(allvar,n.x.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# yv_n_lamb = sy.lambdify(allvar,n.y.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# zv_n_lamb = sy.lambdify(allvar,n.z.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')


# # Root body rootated ref frame
# x_B0_lamb = sy.lambdify(allvar,B0.pos_from(O).to_matrix(n)[0].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# y_B0_lamb = sy.lambdify(allvar,B0.pos_from(O).to_matrix(n)[1].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# z_B0_lamb = sy.lambdify(allvar,B0.pos_from(O).to_matrix(n)[2].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# xv_B0_lamb = sy.lambdify(allvar,b0.x.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# yv_B0_lamb = sy.lambdify(allvar,b0.y.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# zv_B0_lamb = sy.lambdify(allvar,b0.z.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')


# # Robotic Arm 1
# x_B11_lamb = sy.lambdify(allvar,B11.pos_from(O).to_matrix(n)[0].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# y_B11_lamb = sy.lambdify(allvar,B11.pos_from(O).to_matrix(n)[1].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# z_B11_lamb = sy.lambdify(allvar,B11.pos_from(O).to_matrix(n)[2].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# xv_B11_lamb = sy.lambdify(allvar,b11.x.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# yv_B11_lamb = sy.lambdify(allvar,b11.y.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# zv_B11_lamb = sy.lambdify(allvar,b11.z.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')

# x_B21_lamb = sy.lambdify(allvar,B21.pos_from(O).to_matrix(n)[0].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# y_B21_lamb = sy.lambdify(allvar,B21.pos_from(O).to_matrix(n)[1].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# z_B21_lamb = sy.lambdify(allvar,B21.pos_from(O).to_matrix(n)[2].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# xv_B21_lamb = sy.lambdify(allvar,b21.x.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# yv_B21_lamb = sy.lambdify(allvar,b21.y.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# zv_B21_lamb = sy.lambdify(allvar,b21.z.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')

# x_B31_lamb = sy.lambdify(allvar,B31.pos_from(O).to_matrix(n)[0].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# y_B31_lamb = sy.lambdify(allvar,B31.pos_from(O).to_matrix(n)[1].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# z_B31_lamb = sy.lambdify(allvar,B31.pos_from(O).to_matrix(n)[2].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# xv_B31_lamb = sy.lambdify(allvar,b31.x.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# yv_B31_lamb = sy.lambdify(allvar,b31.y.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# zv_B31_lamb = sy.lambdify(allvar,b31.z.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')

# x_B41_lamb = sy.lambdify(allvar,B41.pos_from(O).to_matrix(n)[0].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# y_B41_lamb = sy.lambdify(allvar,B41.pos_from(O).to_matrix(n)[1].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# z_B41_lamb = sy.lambdify(allvar,B41.pos_from(O).to_matrix(n)[2].xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# xv_B41_lamb = sy.lambdify(allvar,b41.x.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# yv_B41_lamb = sy.lambdify(allvar,b41.y.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')
# zv_B41_lamb = sy.lambdify(allvar,b41.z.to_matrix(n).xreplace(gencoord_active_dict).xreplace(substitution), 'numpy')



# Funzione di aggiornamento per l'animazione
def update(frame):
    
    ax.clear()  # Pulisce il grafico precedente
    time_text.set_text(f"t = {t_loaded[frame]:.2f} s")

    # Estrai la nuova posizione dei punti G e B (modifica i calcoli a seconda della tua logica di calcolo delle posizioni)

    # ------------------------------------------------- POINTS EXTRACTIONS ------------------------------------------------------------ #
    # Supponiamo di voler visualizzare alcuni punti, ad esempio O, B0, G01, G11, ...
    # Estrai le posizioni dei punti (le posizioni sono espresse come combinazione di vettori di base)
    
    points = {
        # "O": O.pos_from(O).to_matrix(n)  # O coincide con l'origine, quindi la sua posizione è (0, 0, 0)
        "B0": B0_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "COG": COG_lambd_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),

        # Robotic arm 1
        "G01": G01_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B11": B11_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G11": G11_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B21": B21_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G21": G21_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B31": B31_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G31": G31_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B41": B41_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),

        # Robotic arm 2
        "G02": G02_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B12": B12_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G12": G12_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B22": B22_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G22": G22_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B32": B32_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G32": G32_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B42": B42_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),

        # Robotic arm 3
        "G03": G03_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B13": B13_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G13": G13_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B23": B23_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G23": G23_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B33": B33_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G33": G33_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B43": B43_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),


        # Robotic arm 4
        "G04": G04_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B14": B14_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G14": G14_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B24": B24_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G24": G24_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B34": B34_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "G34": G34_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),
        "B44": B44_lamb_vect(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame]),


        
    }



    # Create a list to store the positions
    positions = []
    colors = []  # Create a list to store colors

    # Estrai le posizioni dei punti e memorizzali come array numpy
    for name, point in points.items():
        # Verifica che il punto sia un array numpy di forma (3, 1)
        if isinstance(point, np.ndarray) and point.shape == (3, 1):
            point = point.flatten()  # Appiattisci l'array in un array 1D di forma (3,)
        
        elif isinstance(point, np.ndarray) and point.shape == (3,): 
            # Se il punto è già in formato corretto, non fare nulla
            pass
        else:
            print(f"Errore nei dati del punto {name}: {point}, Forma: {np.shape(point)}")
            continue  # Salta questo punto se non è formattato correttamente

        # A questo punto 'point' è sempre un array di forma (3,)
        x, y, z = point  # Estrai le coordinate direttamente
        positions.append((x, y, z))
        
        # Assegna colori in base al nome del punto
        if name.startswith('G'):
            colors.append('red')  # Punti con "G" saranno rossi
        elif name.startswith('B'):
            colors.append('blue')  # Punti con "B" saranno blu
        elif name.startswith('COG'):
            colors.append('green')

    # Converti la lista di posizioni in un array numpy
    positions = np.array(positions)


    # Traccia tutti i punti con i relativi colori
    for i, color in enumerate(colors):
        ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], color=color)

    # Annotazione dei punti con i loro nomi
    for i, name in enumerate(points.keys()):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], name, fontsize=10)

    # ---------------------------------------- PLOTTING THE SATELLITE ROBOTIC ARMS ------------------------------------------- #
    # Tracciamo le linee tra i punti (collegando la sequenza che preferisci)
    for i, (point1, point2) in enumerate([("G01", "G11"), ("G11", "G21"), ("G21", "G31"), ("G31", "B41"),
                                          ("G02", "G12"), ("G12", "G22"), ("G22", "G32"), ("G32", "B42"),
                                          ("G03", "G13"), ("G13", "G23"), ("G23", "G33"), ("G33", "B43"),
                                          ("G04", "G14"), ("G14", "G24"), ("G24", "G34"), ("G34", "B44")]):
        
        
        # Estrai le coordinate dei punti
        p1 = points[point1]  # Coordinate punto 1 (array numpy con [x, y, z])
        p2 = points[point2]  # Coordinate punto 2 (array numpy con [x, y, z])

       # Appiattisci gli array se sono di forma (3, 1) per garantirne la forma (3,)
        p1 = p1.flatten()  # Assicuriamoci che p1 abbia la forma (3,)
        p2 = p2.flatten()  # Assicuriamoci che p2 abbia la forma (3,)
        
        # Traccia la linea tra il punto 1 e il punto 2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')  # Linea nera tra i punti

        # -------------------------------------------------------------------------------------------------------------------------- #

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set limits for each axis
        ax.set_xlim([-0.8, 0.8])  # Limit for the X axis
        ax.set_ylim([-0.8, 0.8])  # Limit for the Y axis
        ax.set_zlim([-0.8, 0.8])  # Limit for the Z axis

        

    # # -------------------------------------------- PLOTTING REFERENCE FRAMES ---------------------------------------------------- #

    # # Inertial ref frame
    # x_n_lamb_eval = x_n_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # y_n_lamb_eval = y_n_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # z_n_lamb_eval =  z_n_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # xv_n_lamb_eval = xv_n_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # yv_n_lamb_eval = yv_n_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # zv_n_lamb_eval = zv_n_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])


    # # Root body rootated ref frame
    # x_B0_lamb_eval = x_B0_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # y_B0_lamb_eval = y_B0_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # z_B0_lamb_eval = z_B0_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # xv_B0_lamb_eval = xv_B0_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # yv_B0_lamb_eval = yv_B0_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # zv_B0_lamb_eval =  zv_B0_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])


    # # Robotic Arm 1
    # x_B11_lamb_eval = x_B11_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # y_B11_lamb_eval = y_B11_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # z_B11_lamb_eval = z_B11_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # xv_B11_lamb_eval = xv_B11_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # yv_B11_lamb_eval = yv_B11_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # zv_B11_lamb_eval = zv_B11_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])

    # x_B21_lamb_eval = x_B21_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # y_B21_lamb_eval = y_B21_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # z_B21_lamb_eval = z_B21_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # xv_B21_lamb_eval =  xv_B21_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # yv_B21_lamb_eval =  yv_B21_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # zv_B21_lamb_eval = zv_B21_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])

    # x_B31_lamb_eval = x_B31_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # y_B31_lamb_eval = y_B31_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # z_B31_lamb_eval = z_B31_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # xv_B31_lamb_eval = xv_B31_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # yv_B31_lamb_eval = yv_B31_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # zv_B31_lamb_eval = zv_B31_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])

    # x_B41_lamb_eval = x_B41_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # y_B41_lamb_eval = y_B41_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # z_B41_lamb_eval = z_B41_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # xv_B41_lamb_eval = xv_B41_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # yv_B41_lamb_eval = yv_B41_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])
    # zv_B41_lamb_eval = zv_B41_lamb(q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], q7_sol[frame], q10_sol[frame], q13_sol[frame], q18_sol[frame], q21_sol[frame], q24_sol[frame], q27_sol[frame], q28_sol[frame], q31_sol[frame], q34_sol[frame], q37_sol[frame], q42_sol[frame], q45_sol[frame], q48_sol[frame], q51_sol[frame], q52_sol[frame], q53_sol[frame], q54_sol[frame])







    # # Inertial ref frame
    # ax.quiver(x_n_lamb_eval, y_n_lamb_eval, z_n_lamb_eval, xv_n_lamb_eval[0], xv_n_lamb_eval[1], xv_n_lamb_eval[2], color='red', length = 0.2 ,normalize=True)
    # ax.quiver(x_n_lamb_eval, y_n_lamb_eval, z_n_lamb_eval, yv_n_lamb_eval[0], yv_n_lamb_eval[1], yv_n_lamb_eval[2], color='blue', length = 0.2 ,normalize=True)
    # ax.quiver(x_n_lamb_eval, y_n_lamb_eval, z_n_lamb_eval, zv_n_lamb_eval[0], zv_n_lamb_eval[1], zv_n_lamb_eval[2], color='green', length = 0.2 ,normalize=True)

    # # Root body rotated ref frame
    # ax.quiver(x_B0_lamb_eval, y_B0_lamb_eval, z_B0_lamb_eval, xv_B0_lamb_eval[0], xv_B0_lamb_eval[1], xv_B0_lamb_eval[2], color='red', length = 0.1 ,normalize=True)
    # ax.quiver(x_B0_lamb_eval, y_B0_lamb_eval, z_B0_lamb_eval, yv_B0_lamb_eval[0], yv_B0_lamb_eval[1], yv_B0_lamb_eval[2], color='blue', length = 0.1 ,normalize=True)
    # ax.quiver(x_B0_lamb_eval, y_B0_lamb_eval, z_B0_lamb_eval, zv_B0_lamb_eval[0], zv_B0_lamb_eval[1], zv_B0_lamb_eval[2], color='green', length = 0.1 ,normalize=True)

    # # B11 body ref frame
    # ax.quiver(x_B11_lamb_eval, y_B11_lamb_eval, z_B11_lamb_eval, xv_B11_lamb_eval[0], xv_B11_lamb_eval[1], xv_B11_lamb_eval[2], color='red', length = 0.1 ,normalize=True)
    # ax.quiver(x_B11_lamb_eval, y_B11_lamb_eval, z_B11_lamb_eval, yv_B11_lamb_eval[0], yv_B11_lamb_eval[1], yv_B11_lamb_eval[2], color='blue', length = 0.1 ,normalize=True)
    # ax.quiver(x_B11_lamb_eval, y_B11_lamb_eval, z_B11_lamb_eval, zv_B11_lamb_eval[0], zv_B11_lamb_eval[1], zv_B11_lamb_eval[2], color='green', length = 0.1 ,normalize=True)

    # # B21 body ref frame
    # ax.quiver(x_B21_lamb_eval, y_B21_lamb_eval, z_B21_lamb_eval, xv_B21_lamb_eval[0], xv_B21_lamb_eval[1], xv_B21_lamb_eval[2], color='red', length = 0.1 ,normalize=True)
    # ax.quiver(x_B21_lamb_eval, y_B21_lamb_eval, z_B21_lamb_eval, yv_B21_lamb_eval[0], yv_B21_lamb_eval[1], yv_B21_lamb_eval[2], color='blue', length = 0.1 ,normalize=True)
    # ax.quiver(x_B21_lamb_eval, y_B21_lamb_eval, z_B21_lamb_eval, zv_B21_lamb_eval[0], zv_B21_lamb_eval[1], zv_B21_lamb_eval[2], color='green', length = 0.1 ,normalize=True)

    # # B31 body ref frame
    # ax.quiver(x_B31_lamb_eval, y_B31_lamb_eval, z_B31_lamb_eval, xv_B31_lamb_eval[0], xv_B31_lamb_eval[1], xv_B31_lamb_eval[2], color='red', length = 0.1 ,normalize=True)
    # ax.quiver(x_B31_lamb_eval, y_B31_lamb_eval, z_B31_lamb_eval, yv_B31_lamb_eval[0], yv_B31_lamb_eval[1], yv_B31_lamb_eval[2], color='blue', length = 0.1 ,normalize=True)
    # ax.quiver(x_B31_lamb_eval, y_B31_lamb_eval, z_B31_lamb_eval, zv_B31_lamb_eval[0], zv_B31_lamb_eval[1], zv_B31_lamb_eval[2], color='green', length = 0.1 ,normalize=True)

    # # B41 body ref frame
    # ax.quiver(x_B41_lamb_eval, y_B41_lamb_eval, z_B41_lamb_eval, xv_B41_lamb_eval[0], xv_B41_lamb_eval[1], xv_B41_lamb_eval[2], color='red', length = 0.1 ,normalize=True)
    # ax.quiver(x_B41_lamb_eval, y_B41_lamb_eval, z_B41_lamb_eval, yv_B41_lamb_eval[0], yv_B41_lamb_eval[1], yv_B41_lamb_eval[2], color='blue', length = 0.1 ,normalize=True)
    # ax.quiver(x_B41_lamb_eval, y_B41_lamb_eval, z_B41_lamb_eval, zv_B41_lamb_eval[0], zv_B41_lamb_eval[1], zv_B41_lamb_eval[2], color='green', length = 0.1 ,normalize=True)





'''
# ----------------------------------------------- FORCES ACTING ON OUR BODIES --------------------------------------------------- #
# Forces acting on our bodies
f_11 = F*b11.y
F_11 = (B11, f_11)
modF11 = F_11[1].magnitude().xreplace(gencoord_active_dict).xreplace(substitution)
x_B11 = F_11[0].pos_from(O).to_matrix(n)[0].xreplace(gencoord_active_dict).xreplace(substitution).evalf()
y_B11 = F_11[0].pos_from(O).to_matrix(n)[1].xreplace(gencoord_active_dict).xreplace(substitution).evalf()
z_B11 = F_11[0].pos_from(O).to_matrix(n)[2].xreplace(gencoord_active_dict).xreplace(substitution).evalf()
xF_B11 = F_11[1].to_matrix(n)[0].xreplace(gencoord_active_dict).xreplace(substitution)
yF_B11 = F_11[1].to_matrix(n)[1].xreplace(gencoord_active_dict).xreplace(substitution)
zF_B11 = F_11[1].to_matrix(n)[2].xreplace(gencoord_active_dict).xreplace(substitution)

'''




# ------------------------------------------ PLOTTING FORCES AND TORQUES -------------------------------------------------- #
# Plotting the forces
# Aggiungi il vettore al grafico, con il punto di applicazione (x0, y0, z0) e la direzione (vx, vy, vz)
# ax.quiver(x_B11, y_B11, z_B11, xF_B11, yF_B11, zF_B11, color='green', length = 0.1 ,normalize=True) 


# Crea l'animazione
ani = FuncAnimation(fig, update, frames=len(t_loaded), interval=1, repeat=False)

# Show plot
plt.show()

print('Codice terminato')