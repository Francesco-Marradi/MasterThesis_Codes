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

# Initializing the variables
t = sy.symbols('t')                 # Definition of the time variable
l = sy.symbols('l')                 # Spacecraft bus length [m]
w = sy.symbols('w')                 # Spacecraft bus width [m]
h = sy.symbols('h')                 # Spacecraft bus height [m]
d = sy.symbols('d')                 # Spacecraft bus-arm joint distance [m]
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
F = sy.symbols('F')                 # Force acting on the arms [N]



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
gencoordDer_to_speeds_dict = dict(zip(sy.Matrix([Gen_coord_list]).diff(), sy.Matrix([Gen_speeds_list]))) 

# Dictionary to subsitute the double time derivatives of the gen. coordinates with the time derivative of the gen. speeds
gencoordDerDer_to_speedsDer_dict = dict(zip(sy.Matrix([Gen_coord_list]).diff(t,2), sy.Matrix([Gen_speeds_list]).diff(t)))



#----------------------------------------- REFERENCE FRAME DEFINITION --------------------------------------------------#

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
G01.set_pos(B0, (l/2)*b0.y + d*b0.z)                          # Setting the joint G01 position wrt spacecraft bus center of mass
B11.set_pos(G01, (bl/2)*b11.z)                                # Setting the body B11 position wrt the joint G01
G11.set_pos(B11, (bl/2)*b11.z)                                # Setting the joint G11 position wrt the body B11
B21.set_pos(G11, -(bl/2)*b21.y)                               # Setting the body B21 position wrt the joint G11
G21.set_pos(B21, -(bl/2)*b21.y)                               # Setting the joint G21 position wrt the body B21
B31.set_pos(G21, -(bl/2)*b31.y)                               # Setting the body B31 position wrt the joint G21
G31.set_pos(B31, -(bl/2)*b31.y)                               # Setting the joint G31 position wrt the body B31
B41.set_pos(G31, -(bs/2)*b41.y)                               # Setting the body B31 position wrt the joint G21

# Robotic arm 2
G02.set_pos(B0, (l/2)*b0.y + d*b0.x)                          # Setting the joint G02 position wrt spacecraft bus center of mass
B12.set_pos(G02, (bl/2)*b12.x)                                # Setting the body B12 position wrt the joint G02
G12.set_pos(B12, (bl/2)*b12.x)                                # Setting the joint G12 position wrt the body B12
B22.set_pos(G12, -(bl/2)*b22.y)                               # Setting the body B22 position wrt the joint G12
G22.set_pos(B22, -(bl/2)*b22.y)                               # Setting the joint G22 position wrt the body B22
B32.set_pos(G22, -(bl/2)*b32.y)                               # Setting the body B32 position wrt the joint G22
G32.set_pos(B32, -(bl/2)*b32.y)                               # Setting the joint G32 position wrt the body B32
B42.set_pos(G32, -(bs/2)*b42.y)                               # Setting the body B32 position wrt the joint G22

# Robotic arm 3
G03.set_pos(B0, (l/2)*b0.y - d*b0.z)                          # Setting the joint G03 position wrt spacecraft bus center of mass
B13.set_pos(G03, -(bl/2)*b13.z)                               # Setting the body B13 position wrt the joint G03
G13.set_pos(B13, -(bl/2)*b13.z)                               # Setting the joint G13 position wrt the body B13
B23.set_pos(G13, -(bl/2)*b23.y)                               # Setting the body B23 position wrt the joint G13
G23.set_pos(B23, -(bl/2)*b23.y)                               # Setting the joint G23 position wrt the body B23
B33.set_pos(G23, -(bl/2)*b33.y)                               # Setting the body B33 position wrt the joint G23
G33.set_pos(B33, -(bl/2)*b33.y)                               # Setting the joint G33 position wrt the body B33
B43.set_pos(G33, -(bs/2)*b43.y)                               # Setting the body B33 position wrt the joint G23

# Robotic arm 4
G04.set_pos(B0, (l/2)*b0.y - d*b0.x)                          # Setting the joint G04 position wrt spacecraft bus center of mass
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
b00.orient_body_fixed(n, (q3, q1, q2), 'zxy')

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
    bl: 0.25,               # Robotic arm long piece length [m]
    hl: 0.05,               # Robotic arm long piece height [m]
    wl: 0.05,               # Robotic arm long piece width [m]
    bs: 0.16,               # Robotic arm short piece length [m]
    hs: 0.05,               # Robotic arm short piece height [m]
    ws: 0.05,               # Robotic arm short piece width [m]
    mlongarm: 0.836,        # Robotic Long arm piece mass [kg]
    mshortarm: 0.540,       # Robotic Short arm piece mass [kg]
    mbus: 122.8,            # Spacecraft bus mass [kg]

    T12 : 0.0,              # Torque acting on the arms [N*m]
    T22 : 0.0,              # Torque acting on the arms [N*m]
    T32 : 0.0,              # Torque acting on the arms [N*m]
    T42 : 0.0,              # Torque acting on the arms [N*m]
    T13 : 0.0,              # Torque acting on the arms [N*m]
    T23 : 0.0,              # Torque acting on the arms [N*m]
    T33 : 0.0,              # Torque acting on the arms [N*m]
    T43 : 0.0,              # Torque acting on the arms [N*m]
    T14 : 0.0,              # Torque acting on the arms [N*m]
    T24 : 0.0,              # Torque acting on the arms [N*m]
    T34 : 0.0,              # Torque acting on the arms [N*m]
    T44 : 0.0,              # Torque acting on the arms [N*m]
    F : 0.0,                # Force acting on the arms [N]
}



# ------------------------------------------ LOADING DATA FOR ANIMATION -------------------------------------------------- #
# Caricamento dei dati dal file JSON
with open('solution_1arm_torqueapplied.json', 'r') as f:
    data_loaded = json.load(f)

# Estrazione dei dati dal JSON
t_loaded = np.array(data_loaded["t"])  # Converti di nuovo in numpy array
y_loaded = np.array(data_loaded["y"])

# Unpacking the datas from the vector y_loaded
u1_sol = y_loaded[0,:]
u2_sol = y_loaded[1,:]
u3_sol = y_loaded[2,:]
u4_sol = y_loaded[3,:]
u7_sol = y_loaded[4,:]
u10_sol = y_loaded[5,:]
u13_sol = y_loaded[6,:]
u52_sol = y_loaded[7,:]
u53_sol = y_loaded[8,:]
u54_sol = y_loaded[9,:]

q1_sol = y_loaded[10,:] 
q2_sol = y_loaded[11,:]
q3_sol = y_loaded[12,:]
q4_sol = y_loaded[13,:] 
q7_sol = y_loaded[14,:]
q10_sol = y_loaded[15,:]
q13_sol = y_loaded[16,:]
q52_sol = y_loaded[17,:] 
q53_sol = y_loaded[18,:]
q54_sol = y_loaded[19,:]


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

# Funzione di aggiornamento per l'animazione
def update(frame):
    
    ax.clear()  # Pulisce il grafico precedente
    time_text.set_text(f"t = {t_loaded[frame]:.2f} s")

    # Calcola le nuove posizioni dei punti in base al frame
    gencoord_active = sy.Matrix([q1_sol[frame], q2_sol[frame], q3_sol[frame], q4_sol[frame], 0, 0, q7_sol[frame], 0, 0, q10_sol[frame], 0, 0, q13_sol[frame], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, q52_sol[frame], q53_sol[frame], q54_sol[frame]])  # Aggiorna le coordinate
    genspeeds_active = sy.Matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Constructing the dictionaries
    gencoord_active_dict = dict(zip(Gen_coord_list ,gencoord_active))
    genspeeds_active_dict = dict(zip(Gen_speeds_list ,genspeeds_active))

    # Estrai la nuova posizione dei punti G e B (modifica i calcoli a seconda della tua logica di calcolo delle posizioni)

    # ------------------------------------------------- POINTS EXTRACTIONS ------------------------------------------------------------ #
    # Supponiamo di voler visualizzare alcuni punti, ad esempio O, B0, G01, G11, ...
    # Estrai le posizioni dei punti (le posizioni sono espresse come combinazione di vettori di base)
    points = {
        # "O": O.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),  # O coincide con l'origine, quindi la sua posizione è (0, 0, 0)
        "B0": B0.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),

        # Robotic arm 1
        "G01": G01.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B11": B11.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G11": G11.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B21": B21.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G21": G21.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B31": B31.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G31": G31.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B41": B41.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),

        # Robotic arm 2
        "G02": G02.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B12": B12.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G12": G12.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B22": B22.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G22": G22.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B32": B32.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G32": G32.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B42": B42.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),

        # Robotic arm 3
        "G03": G03.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B13": B13.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G13": G13.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B23": B23.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G23": G23.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B33": B33.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G33": G33.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B43": B43.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),

        # Robotic arm 4
        "G04": G04.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B14": B14.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G14": G14.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B24": B24.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G24": G24.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B34": B34.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "G34": G34.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),
        "B44": B44.pos_from(O).to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf(),

        
    }

    # Create a list to store the positions
    positions = []
    colors = []  # Create a list to store colors

    # Extract the positions of each point and store them in a list as numpy arrays
    for name, point in points.items():
        # Extract coordinates (x, y, z)
        coordinates = point.args[2]  # point.args[2] contains the tuple/list (x, y, z)
        x, y, z = coordinates  # Unpack the values
        positions.append((x, y, z))
        
        # Assign colors based on the point name
        if name.startswith('G'):
            colors.append('red')  # Points starting with "G" will be red
        elif name.startswith('B'):
            colors.append('blue')  # Points starting with "B" will be blue

    # Convert to a numpy array for easier plotting
    positions = np.array(positions)

    # Plot all points with their corresponding colors
    for i, color in enumerate(colors):
        ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], color=color)

    # Annotate points with their names
    for i, name in enumerate(points.keys()):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], name, fontsize=10)



    # ---------------------------------------- PLOTTING THE SATELLITE ROBOTIC ARMS ------------------------------------------- #
    # Plotting the lines to connect the points
    # Add lines between dots (connecting the sequence you prefer).
    for i, (point1, point2) in enumerate([("G01", "G11"), ("G11", "G21"), ("G21", "G31"), ("G31", "B41"),
                                          ("G02", "G12"), ("G12", "G22"), ("G22", "G32"), ("G32", "B42"),
                                          ("G03", "G13"), ("G13", "G23"), ("G23", "G33"), ("G33", "B43"),
                                          ("G04", "G14"), ("G14", "G24"), ("G24", "G34"), ("G34", "B44")]):
        # Estrai le coordinate dei punti
        p1 = points[point1].args[2]  # Coordinates point 1
        p2 = points[point2].args[2]  # Coordinates point 2
        
        # Draw the line between point 1 and point 2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')  # Linea nera tra i punti



        # -------------------------------------------------------------------------------------------------------------------------- #

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set limits for each axis
        ax.set_xlim([-1.0, 1.0])  # Limit for the X axis
        ax.set_ylim([-1.0, 1.0])  # Limit for the Y axis
        ax.set_zlim([-1.0, 1.0])  # Limit for the Z axis





'''
# ----------------------------------------------- FORCES ACTING ON OUR BODIES --------------------------------------------------- #
# Forces acting on our bodies
f_11 = F*b11.y
F_11 = (B11, f_11)
modF11 = F_11[1].magnitude().xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
x_B11 = F_11[0].pos_from(O).to_matrix(n)[0].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
y_B11 = F_11[0].pos_from(O).to_matrix(n)[1].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
z_B11 = F_11[0].pos_from(O).to_matrix(n)[2].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
xF_B11 = F_11[1].to_matrix(n)[0].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
yF_B11 = F_11[1].to_matrix(n)[1].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
zF_B11 = F_11[1].to_matrix(n)[2].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)

'''
'''
# ------------------------------------------------- REFERENCE FRAMES ------------------------------------------------------------ #
# Inertial ref frame
x_n = O.pos_from(O).to_matrix(n)[0].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
y_n = O.pos_from(O).to_matrix(n)[1].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
z_n = O.pos_from(O).to_matrix(n)[2].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
xv_n = n.x.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
yv_n = n.y.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
zv_n = n.z.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)


# Root body rootated ref frame
x_B0 = B0.pos_from(O).to_matrix(n)[0].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
y_B0 = B0.pos_from(O).to_matrix(n)[1].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
z_B0 = B0.pos_from(O).to_matrix(n)[2].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
xv_B0 = b0.x.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
yv_B0 = b0.y.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
zv_B0 = b0.z.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)


# Robotic Arm 1
x_B11 = B11.pos_from(O).to_matrix(n)[0].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
y_B11 = B11.pos_from(O).to_matrix(n)[1].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
z_B11 = B11.pos_from(O).to_matrix(n)[2].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
xv_B11 = b11.x.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
yv_B11 = b11.y.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
zv_B11 = b11.z.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)

x_B21 = B21.pos_from(O).to_matrix(n)[0].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
y_B21 = B21.pos_from(O).to_matrix(n)[1].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
z_B21 = B21.pos_from(O).to_matrix(n)[2].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
xv_B21 = b21.x.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
yv_B21 = b21.y.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
zv_B21 = b21.z.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)

x_B31 = B31.pos_from(O).to_matrix(n)[0].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
y_B31 = B31.pos_from(O).to_matrix(n)[1].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
z_B31 = B31.pos_from(O).to_matrix(n)[2].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
xv_B31 = b31.x.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
yv_B31 = b31.y.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
zv_B31 = b31.z.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)

x_B41 = B41.pos_from(O).to_matrix(n)[0].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
y_B41 = B41.pos_from(O).to_matrix(n)[1].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
z_B41 = B41.pos_from(O).to_matrix(n)[2].xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution).evalf()
xv_B41 = b41.x.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
yv_B41 = b41.y.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)
zv_B41 = b41.z.to_matrix(n).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)


'''





# ------------------------------------------ PLOTTING FORCES AND TORQUES -------------------------------------------------- #
# Plotting the forces
# Aggiungi il vettore al grafico, con il punto di applicazione (x0, y0, z0) e la direzione (vx, vy, vz)
# ax.quiver(x_B11, y_B11, z_B11, xF_B11, yF_B11, zF_B11, color='green', length = 0.1 ,normalize=True) 


'''
# -------------------------------------------- PLOTTING REFERENCE FRAMES ---------------------------------------------------- #

# Inertial ref frame
ax.quiver(x_n, y_n, z_n, xv_n[0], xv_n[1], xv_n[2], color='red', length = 0.2 ,normalize=True)
ax.quiver(x_n, y_n, z_n, yv_n[0], yv_n[1], yv_n[2], color='blue', length = 0.2 ,normalize=True)
ax.quiver(x_n, y_n, z_n, zv_n[0], zv_n[1], zv_n[2], color='green', length = 0.2 ,normalize=True)

# Root body rotated ref frame
ax.quiver(x_B0, y_B0, z_B0, xv_B0[0], xv_B0[1], xv_B0[2], color='red', length = 0.1 ,normalize=True)
ax.quiver(x_B0, y_B0, z_B0, yv_B0[0], yv_B0[1], yv_B0[2], color='blue', length = 0.1 ,normalize=True)
ax.quiver(x_B0, y_B0, z_B0, zv_B0[0], zv_B0[1], zv_B0[2], color='green', length = 0.1 ,normalize=True)

# B11 body ref frame
ax.quiver(x_B11, y_B11, z_B11, xv_B11[0], xv_B11[1], xv_B11[2], color='red', length = 0.1 ,normalize=True)
ax.quiver(x_B11, y_B11, z_B11, yv_B11[0], yv_B11[1], yv_B11[2], color='blue', length = 0.1 ,normalize=True)
ax.quiver(x_B11, y_B11, z_B11, zv_B11[0], zv_B11[1], zv_B11[2], color='green', length = 0.1 ,normalize=True)

# B21 body ref frame
ax.quiver(x_B21, y_B21, z_B21, xv_B21[0], xv_B21[1], xv_B21[2], color='red', length = 0.1 ,normalize=True)
ax.quiver(x_B21, y_B21, z_B21, yv_B21[0], yv_B21[1], yv_B21[2], color='blue', length = 0.1 ,normalize=True)
ax.quiver(x_B21, y_B21, z_B21, zv_B21[0], zv_B21[1], zv_B21[2], color='green', length = 0.1 ,normalize=True)

# B31 body ref frame
ax.quiver(x_B31, y_B31, z_B31, xv_B31[0], xv_B31[1], xv_B31[2], color='red', length = 0.1 ,normalize=True)
ax.quiver(x_B31, y_B31, z_B31, yv_B31[0], yv_B31[1], yv_B31[2], color='blue', length = 0.1 ,normalize=True)
ax.quiver(x_B31, y_B31, z_B31, zv_B31[0], zv_B31[1], zv_B31[2], color='green', length = 0.1 ,normalize=True)

# B41 body ref frame
ax.quiver(x_B41, y_B41, z_B41, xv_B41[0], xv_B41[1], xv_B41[2], color='red', length = 0.1 ,normalize=True)
ax.quiver(x_B41, y_B41, z_B41, yv_B41[0], yv_B41[1], yv_B41[2], color='blue', length = 0.1 ,normalize=True)
ax.quiver(x_B41, y_B41, z_B41, zv_B41[0], zv_B41[1], zv_B41[2], color='green', length = 0.1 ,normalize=True)
'''


# Crea l'animazione
ani = FuncAnimation(fig, update, frames=len(t_loaded), interval=50, repeat=False)

# Show plot
plt.show()

print('Codice terminato')