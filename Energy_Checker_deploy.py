
# ------------------------------------ IMPORTING THE NECESSARY PACKAGES ----------------------------------------------- #

# Importing only the system package to tell python in which other folders search the other packages
import sys
import os
import matplotlib.pyplot as plt

# To find the files with the functions, we need to tell python to "go up two folders" from the folder in which this file is 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# Importing the other packages
from utility.import_packages import sy, me, time, json, np, itertools

# Importing the functions
from utility import matrix_kane_functions as mk
from utility.miscellaneous_functions import export_matrices_to_json


# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

#Cleaning the terminal 
os.system('cls')


print('ciao 123')

# ---------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

start = time.perf_counter()



#---------------------------------------- SYSTEM GENERAL PARAMETERS -------------------------------------------------#   
print('Loading symbols, generalized coordinates and speeds...')


# Topological tree characteristics
nb = 17                                 # Number of bodies present 
nbranch = 4                             # Number of branches
n_bodies_in_branch = 5                           # Number of bodies in each branch
njoint_in_branch = 4                    # Number of joints in each branch
n_joints = nbranch*njoint_in_branch     # Number of joints presents


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
k12 = sy.symbols('k12')             # B12 body spring's stiffness [N*m/rad]
k22 = sy.symbols('k22')             # B22 body spring's stiffness [N*m/rad] 
k32 = sy.symbols('k32')             # B32 body spring's stiffness [N*m/rad] 
k42 = sy.symbols('k42')             # B42 body spring's stiffness [N*m/rad] 
k13 = sy.symbols('k13')             # B13 body spring's stiffness [N*m/rad]
k23 = sy.symbols('k23')             # B23 body spring's stiffness [N*m/rad] 
k33 = sy.symbols('k33')             # B33 body spring's stiffness [N*m/rad] 
k43 = sy.symbols('k43')             # B43 body spring's stiffness [N*m/rad] 
k14 = sy.symbols('k14')             # B14 body spring's stiffness [N*m/rad]
k24 = sy.symbols('k24')             # B24 body spring's stiffness [N*m/rad] 
k34 = sy.symbols('k34')             # B34 body spring's stiffness [N*m/rad] 
k44 = sy.symbols('k44')             # B44 body spring's stiffness [N*m/rad] 

c11 = sy.symbols('c11')             # B11 body damping coefficient [(N*m/s)/(rad/s)]
c21 = sy.symbols('c21')             # B21 body damping coefficient [(N*m/s)/(rad/s)]
c31 = sy.symbols('c31')             # B31 body damping coefficient [(N*m/s)/(rad/s)]
c41 = sy.symbols('c41')             # B41 body damping coefficient [(N*m/s)/(rad/s)]
c12 = sy.symbols('c12')             # B12 body damping coefficient [(N*m/s)/(rad/s)]
c22 = sy.symbols('c22')             # B22 body damping coefficient [(N*m/s)/(rad/s)]
c32 = sy.symbols('c32')             # B32 body damping coefficient [(N*m/s)/(rad/s)]
c42 = sy.symbols('c42')             # B42 body damping coefficient [(N*m/s)/(rad/s)]
c13 = sy.symbols('c13')             # B13 body damping coefficient [(N*m/s)/(rad/s)]
c23 = sy.symbols('c23')             # B23 body damping coefficient [(N*m/s)/(rad/s)]
c33 = sy.symbols('c33')             # B33 body damping coefficient [(N*m/s)/(rad/s)]
c43 = sy.symbols('c43')             # B43 body damping coefficient [(N*m/s)/(rad/s)]
c14 = sy.symbols('c14')             # B14 body damping coefficient [(N*m/s)/(rad/s)]
c24 = sy.symbols('c24')             # B24 body damping coefficient [(N*m/s)/(rad/s)]
c34 = sy.symbols('c34')             # B34 body damping coefficient [(N*m/s)/(rad/s)]
c44 = sy.symbols('c44')             # B44 body damping coefficient [(N*m/s)/(rad/s)]



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

# Generalized coordinates and speeds automatic definition
n_var, gen_coord_list, gen_speed_list, coorddot_to_speed_dict  = mk.define_gen_coord_and_speed(n_joints)

# We need also to manually define them outside of a list, so they can be  used to define ref. systems and forces
for i in range(n_var):
    exec(f'q{i+1} = me.dynamicsymbols(f"q{i+1}")')
    exec(f'u{i+1} = me.dynamicsymbols(f"u{i+1}")')



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



# ---------------------------------------------- DEFINING THE CONSTANT VALUES ------------------------------------------------- #
print('Defining and substituting the constant values into the equations...')

# Defining the dictionary for the substitution
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
    mlongarm: 0.510,        # Robotic Long arm piece mass [kg]
    mshortarm: 0.320,       # Robotic Short arm piece mass [kg]
    mbus: 121.2,            # Spacecraft bus mass [kg]
}

# Defining the spring coefficients [Nm/rad]
k_list = [0, 0.52,0.33,0.22,0.11,0.52,0.33,0.22,0.11,0.52,0.33,0.22,0.11,0.52,0.33,0.22,0.11]

# Defining the spring coefficients [(N*m/s)/(rad/s)]
c_list = [0, 1.15,0.75,0.55,0.35,1.15,0.75,0.55,0.35,1.15,0.75,0.55,0.35,1.15,0.75,0.55,0.35]



# ----------------------------------------------- TURNING OFF DOFs -------------------------------------------------------------- #

# We will use a dictionary that will set to zero each gen. coordinates/speeds, blocking that dof for the system.   [4 ARMS ACTIVE]
                             #x   y  z   x   y  z  x   y  z   x   y  z   x   y  z  x  y  z    x  y   z   x  y   z   x  y  z    x    y  z   x   y  z   x   y  z   x   y  z  x  y  z    x  y   z   x  y   z   x  y  z     x   y    z
gencoord_active = sy.Matrix([q1, q2, q3, q4, 0, 0, q7, 0, 0, q10, 0, 0, q13, 0, 0, 0, 0, q18, 0, 0, q21, 0, 0, q24, 0, 0, q27, q28, 0, 0, q31, 0, 0, q34, 0, 0, q37, 0, 0, 0, 0, q42, 0, 0, q45, 0, 0, q48, 0, 0, q51, q52, q53, q54])
genspeeds_active = sy.Matrix([u1, u2, u3, u4, 0, 0, u7, 0, 0, u10, 0, 0, u13, 0, 0, 0, 0, u18, 0, 0, u21, 0, 0, u24, 0, 0, u27, u28, 0, 0, u31, 0, 0, u34, 0, 0, u37, 0, 0, 0, 0, u42, 0, 0, u45, 0, 0, u48, 0, 0, u51, u52, u53, u54])

# # We will use a dictionary that will set to zero each gen. coordinates/speeds, blocking that dof for the system.   [1 ARM ACTIVE - ONLY THE B11,B21,B31,B41 - SPACECRAFT BUS FIXED]
# gencoord_active = sy.Matrix([0, 0, 0, q4, 0, 0, q7, 0, 0, q10, 0, 0, q13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# genspeeds_active = sy.Matrix([0, 0, 0, u4, 0, 0, u7, 0, 0, u10, 0, 0, u13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# # We will use a dictionary that will set to zero each gen. coordinates/speeds, blocking that dof for the system.   [1 ARM ACTIVE - ONLY THE B11,B21,B31,B41]
# gencoord_active = sy.Matrix([q1, q2, q3, q4, 0, 0, q7, 0, 0, q10, 0, 0, q13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, q52, q53, q54])
# genspeeds_active = sy.Matrix([u1, u2, u3, u4, 0, 0, u7, 0, 0, u10, 0, 0, u13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u52, u53, u54])


# Dictionary to turn off the dof of the system
gencoord_active_dict = dict(zip(gen_coord_list ,gencoord_active))
genspeeds_active_dict = dict(zip(gen_speed_list ,genspeeds_active))



# --------------------------------------------- MASS AND INERTIA MATRICES ------------------------------------------------------- #
print('Defining the inertia matrices and mass values...')

# Inertia Dyadic for each body, centered in their respective com and expressed along their principal inertia axis

# Since we defined a body ref. frame centered in each body com and alligned with its principal inertia axis, we can
# easily define the inertia matrix for each body using the outer product:

# Spacecraft bus
I_0 = (1/12)*mbus*((l**2 + h**2)*me.outer(b00.x, b00.x) + (h**2 + w**2)*me.outer(b00.y, b00.y) + (l**2 + w**2)*me.outer(b00.z, b00.z))

# Robotic arm 1
I_11 = (1/12)*mlongarm*((hl**2 + bl**2)*me.outer(b11.x, b11.x) + (wl**2 + bl**2)*me.outer(b11.y, b11.y) + (hl**2 + wl**2)*me.outer(b11.z, b11.z))
I_21 = (1/12)*mlongarm*((hl**2 + bl**2)*me.outer(b21.x, b21.x) + (hl**2 + wl**2)*me.outer(b21.y, b21.y) + (bl**2 + wl**2)*me.outer(b21.z, b21.z))
I_31 = (1/12)*mlongarm*((hl**2 + bl**2)*me.outer(b31.x, b31.x) + (hl**2 + wl**2)*me.outer(b31.y, b31.y) + (bl**2 + wl**2)*me.outer(b31.z, b31.z))
I_41 = (1/12)*mshortarm*((hs**2 + bs**2)*me.outer(b41.x, b41.x) + (hs**2 + ws**2)*me.outer(b41.y, b41.y) + (bs**2 + ws**2)*me.outer(b41.z, b41.z))

# Robotic arm 2
I_12 = (1/12)*mlongarm*((hl**2 + wl**2)*me.outer(b12.x, b12.x) + (wl**2 + bl**2)*me.outer(b12.y, b12.y) + (bl**2 + hl**2)*me.outer(b12.z, b12.z))
I_22 = (1/12)*mlongarm*((wl**2 + bl**2)*me.outer(b22.x, b22.x) + (hl**2 + wl**2)*me.outer(b22.y, b22.y) + (bl**2 + hl**2)*me.outer(b22.z, b22.z))
I_32 = (1/12)*mlongarm*((wl**2 + bl**2)*me.outer(b32.x, b32.x) + (hl**2 + wl**2)*me.outer(b32.y, b32.y) + (bl**2 + hl**2)*me.outer(b32.z, b32.z))
I_42 = (1/12)*mshortarm*((ws**2 + bs**2)*me.outer(b42.x, b42.x) + (hs**2 + ws**2)*me.outer(b42.y, b42.y) + (bs**2 + hs**2)*me.outer(b42.z, b42.z))

# Robotic arm 3
I_13 = (1/12)*mlongarm*((hl**2 + bl**2)*me.outer(b13.x, b13.x) + (wl**2 + bl**2)*me.outer(b13.y, b13.y) + (hl**2 + wl**2)*me.outer(b13.z, b13.z))
I_23 = (1/12)*mlongarm*((hl**2 + bl**2)*me.outer(b23.x, b23.x) + (hl**2 + wl**2)*me.outer(b23.y, b23.y) + (bl**2 + wl**2)*me.outer(b23.z, b23.z))
I_33 = (1/12)*mlongarm*((hl**2 + bl**2)*me.outer(b33.x, b33.x) + (hl**2 + wl**2)*me.outer(b33.y, b33.y) + (bl**2 + wl**2)*me.outer(b33.z, b33.z))
I_43 = (1/12)*mshortarm*((hs**2 + bs**2)*me.outer(b43.x, b43.x) + (hs**2 + ws**2)*me.outer(b43.y, b43.y) + (bs**2 + ws**2)*me.outer(b43.z, b43.z))

# Robotic arm 4
I_14 = (1/12)*mlongarm*((hl**2 + wl**2)*me.outer(b14.x, b14.x) + (wl**2 + bl**2)*me.outer(b14.y, b14.y) + (bl**2 + hl**2)*me.outer(b14.z, b14.z))
I_24 = (1/12)*mlongarm*((wl**2 + bl**2)*me.outer(b24.x, b24.x) + (hl**2 + wl**2)*me.outer(b24.y, b24.y) + (bl**2 + hl**2)*me.outer(b24.z, b24.z))
I_34 = (1/12)*mlongarm*((wl**2 + bl**2)*me.outer(b34.x, b34.x) + (hl**2 + wl**2)*me.outer(b34.y, b34.y) + (bl**2 + hl**2)*me.outer(b34.z, b34.z))
I_44 = (1/12)*mshortarm*((ws**2 + bs**2)*me.outer(b44.x, b44.x) + (hs**2 + ws**2)*me.outer(b44.y, b44.y) + (bs**2 + hs**2)*me.outer(b44.z, b44.z))


# Putting everything inside a list and then these lists inside another list

# Spacecraft bus
I_list_bus = [I_0.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict)]

# Robotic arm 1
I_list_arm1 = [I_11.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_21.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_31.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_41.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict)]

# Robotic arm 2
I_list_arm2 = [I_12.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_22.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_32.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_42.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict)]

# Robotic arm 3
I_list_arm3 = [I_13.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_23.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_33.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_43.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict)]

# Robotic arm 4
I_list_arm4 = [I_14.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_24.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_34.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict), I_44.to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict)]


# Inertia List
I_list = [I_list_bus, I_list_arm1, I_list_arm2, I_list_arm3, I_list_arm4]

# --------------------------------------- #

# Creating a list containing the mass values

# Spacecraft bus
M_list_bus = [mbus]

# Robotic arm 1
M_list_arm1 = [mlongarm, mlongarm, mlongarm, mshortarm]

# Robotic arm 2
M_list_arm2 = [mlongarm, mlongarm, mlongarm, mshortarm]

# Robotic arm 3
M_list_arm3 = [mlongarm, mlongarm, mlongarm, mshortarm]

# Robotic arm 4
M_list_arm4 = [mlongarm, mlongarm, mlongarm, mshortarm]

# Mass list
M_list = [M_list_bus, M_list_arm1, M_list_arm2, M_list_arm3, M_list_arm4]



# ---------------------------------------- COMPUTING THE VELOCITY VECTORS FOR EACH BODY --------------------------------------- # 
print('Computing the velocity vectors for each body...')

# Here we are computing the Velocity vectors, both angular and linear of each body composing our system. We are also substituting
# each time derivative of the gen. coordinates with the corresponding gen. speeds

# Spacecraft bus Angular velocity vector
omegabody_bus = b00.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)


# Robotic arms angular velocity vectors computation 
# Robotic arm 1
omegabody_list_Arm1 = [b11.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b21.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b31.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b41.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)]

# Robotic arm 2
omegabody_list_Arm2 = [b12.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b22.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b32.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b42.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)]

# Robotic arm 3
omegabody_list_Arm3 = [b13.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b23.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b33.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b43.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)]

# Robotic arm 4
omegabody_list_Arm4 = [b14.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b24.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b34.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), b44.ang_vel_in(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)]


# Putting all lists inside an other list
omegabody_list = [omegabody_list_Arm1, omegabody_list_Arm2, omegabody_list_Arm3, omegabody_list_Arm4]


#---------------------------------------------#

# Spacecraft bus Linear velocity vector
vbody_bus = B0.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)


# Linear robotic arms velocity vectors computation 
# Robotic arm 1
vbody_list_Arm1 = [B11.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B21.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B31.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B41.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)]

# Robotic arm 2
vbody_list_Arm2 = [B12.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B22.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B32.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B42.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)]

# Robotic arm 3
vbody_list_Arm3 = [B13.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B23.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B33.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B43.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)]

# Robotic arm 4
vbody_list_Arm4 = [B14.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B24.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B34.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution), B44.vel(n).to_matrix(n).xreplace(coorddot_to_speed_dict).xreplace(genspeeds_active_dict).xreplace(gencoord_active_dict).xreplace(substitution)]


# Putting all lists inside an other list
vbody_list = [vbody_list_Arm1, vbody_list_Arm2, vbody_list_Arm3, vbody_list_Arm4]



# --------------------------------------------- ADJUSTING UP THE LISTS --------------------------------------------------- #
print('Adjusting up...')

# Creating a "long list" for: angular and linear velocity vectors and also inertia and masses

# Angular velocity vectors "lists of lists" to "very long list"
omegabody_longlist  = [omegabody_bus] + list(itertools.chain.from_iterable(omegabody_list))

# Linear velocity vectors "lists of lists" to "very long list"
vbody_longlist =  [vbody_bus] + list(itertools.chain.from_iterable(vbody_list))

# Inertia values list
I_longlist = list(itertools.chain(*I_list))

# Mass values list
M_longlist = list(itertools.chain(*M_list))

# Angle list, to compute dissipative work
An_list = [0, q4, q7, q10, q13, q18, q21, q24, q27, q28, q31, q34, q37, q42, q45, q48, q51]
An_rate_list = [0, u4, u7, u10, u13, u18, u21, u24, u27, u28, u31, u34, u37, u42, u45, u48, u51]

# Defining the angular displacement of each spring, to compute their potential energy
spring_disp = [0, q4_deploy - q4, q7_deploy - q7, q10_deploy - q10, q13_deploy - q13, q18_deploy - q18, q21_deploy - q21, q24_deploy - q24, q27_deploy - q27,
               q28_deploy - q28, q31_deploy - q31, q34_deploy - q34, q37_deploy - q37, q42_deploy - q42, q45_deploy - q45, q48_deploy - q48, q51_deploy - q51]


# Next step is to substitute the constant values in the angular and linear velocity vectors, then use the lambdify function
# on the lists to increase the calculations performances
I_longlist         = [expr.xreplace(substitution) for expr in I_longlist]
M_longlist         = [expr.xreplace(substitution) for expr in M_longlist]


all_symbolic_var = (u1,u2,u3,u4,u7,u10,u13,u18,u21,u24,u27,u28,u31,u34,u37,u42,u45,u48,u51,u52,u53,u54,q1,q2,q3,q4,q7,q10,q13,q18,q21,q24,q27,q28,q31,q34,q37,q42,q45,q48,q51,q52,q53,q54)
omegabody_longlist_lambd = [sy.lambdify(all_symbolic_var, expr, 'numpy') for expr in omegabody_longlist]
vbody_longlist_lambd = [sy.lambdify(all_symbolic_var, expr, 'numpy') for expr in vbody_longlist]
I_longlist_lambd = [sy.lambdify(all_symbolic_var, expr, 'numpy') for expr in I_longlist]
An_list_lambd = [sy.lambdify(all_symbolic_var, expr, 'numpy') for expr in An_list]      
An_rate_list_lambd = [sy.lambdify(all_symbolic_var, expr, 'numpy') for expr in An_rate_list]                               
spring_disp_lambd = [sy.lambdify(all_symbolic_var, expr, 'numpy') for expr in spring_disp]



# ---------------------------------------------- LOADING THE FILES ------------------------------------------------- #
print('Loading the numerical integrator results...')

# Loading the dynamic system state changes during deployment 

# Reading the Json solver file
with open('solution_4arms_Preloadedsprings_dampers_FINAL.json', 'r') as json_file:
    data_craft = json.load(json_file)


# ----------------------------------------------------------- #

# Extracting the solver data from the JSON

# Extracting time
t_loaded = np.array(data_craft['t'])  


# Extracting the bus angular velocities
u1_data = np.array(data_craft['y'][0])
u2_data = np.array(data_craft['y'][1])
u3_data = np.array(data_craft['y'][2])

# Extracting robotic arm 1 angular velocities
u4_data = np.array(data_craft['y'][3])
u7_data = np.array(data_craft['y'][4])
u10_data = np.array(data_craft['y'][5])
u13_data = np.array(data_craft['y'][6])

# Extracting robotic arm 2 angular velocities
u18_data = np.array(data_craft['y'][7])
u21_data = np.array(data_craft['y'][8])
u24_data = np.array(data_craft['y'][9])
u27_data = np.array(data_craft['y'][10])

# Extracting robotic arm 3 angular velocities
u28_data = np.array(data_craft['y'][11])
u31_data = np.array(data_craft['y'][12])
u34_data = np.array(data_craft['y'][13])
u37_data = np.array(data_craft['y'][14])

# Extracting robotic arm 4 angular velocities
u42_data = np.array(data_craft['y'][15])
u45_data = np.array(data_craft['y'][16])
u48_data = np.array(data_craft['y'][17])
u51_data = np.array(data_craft['y'][18])

# Extracting the bus linear velocities
u52_data = np.array(data_craft['y'][19])
u53_data = np.array(data_craft['y'][20])
u54_data = np.array(data_craft['y'][21])


# Extracting the bus angular position
q1_data = np.array(data_craft['y'][22])
q2_data = np.array(data_craft['y'][23])
q3_data = np.array(data_craft['y'][24])

# Extracting robotic arm 1 angular position
q4_data = np.array(data_craft['y'][25])
q7_data = np.array(data_craft['y'][26])
q10_data = np.array(data_craft['y'][27])
q13_data = np.array(data_craft['y'][28])

# Extracting robotic arm 2 angular position
q18_data = np.array(data_craft['y'][29])
q21_data = np.array(data_craft['y'][30])
q24_data = np.array(data_craft['y'][31])
q27_data = np.array(data_craft['y'][32])

# Extracting robotic arm 3 angular position
q28_data = np.array(data_craft['y'][33])
q31_data = np.array(data_craft['y'][34])
q34_data = np.array(data_craft['y'][35])
q37_data = np.array(data_craft['y'][36])

# Extracting robotic arm 4 angular position
q42_data = np.array(data_craft['y'][37])
q45_data = np.array(data_craft['y'][38])
q48_data = np.array(data_craft['y'][39])
q51_data = np.array(data_craft['y'][40])

# Extracting the bus linear position
q52_data = np.array(data_craft['y'][41])
q53_data = np.array(data_craft['y'][42])
q54_data = np.array(data_craft['y'][43])



# ---------------------------------------------- COMPUTING THE SYSTEM ENERGY ------------------------------------------------- #
print('Computing the System Total Energy...')

# Here we are going to compute the system total energy, by summing the total energy of each body composing the system. This can be done
# by computing the angular and linear velocity vectors with respect the same reference frame, thats is, for us, the inertial ref. frame 

# Defining the empty list that will contain the system total energy values
Etot = [0]*len(t_loaded)
Emecc = [0]*len(t_loaded)
Ecin = [0]*len(t_loaded)
Ecin_lin = [0]*len(t_loaded)
Ecin_rot = [0]*len(t_loaded)
Epot = [0]*len(t_loaded)
W_diss_in = [0]*len(t_loaded)
W_diss_tot = [0]*len(t_loaded)



# Defining a for cycle to compute the Kinetic Energy of each body: E_kin = E_kin_traslational + E_kin_rotational
for i in range (len(t_loaded)):
    
    for j in range (len(M_longlist)):
        

        # Computing the quantities for each body at this time step
        vbody = vbody_longlist_lambd[j](
            u1_data[i],u2_data[i],u3_data[i],u4_data[i],u7_data[i],u10_data[i],u13_data[i],u18_data[i],u21_data[i],u24_data[i],u27_data[i],u28_data[i],u31_data[i]
            ,u34_data[i],u37_data[i],u42_data[i],u45_data[i],u48_data[i],u51_data[i],u52_data[i],u53_data[i],u54_data[i]
            ,q1_data[i],q2_data[i],q3_data[i],q4_data[i],q7_data[i],q10_data[i],q13_data[i],q18_data[i],q21_data[i],q24_data[i],q27_data[i],q28_data[i],q31_data[i]
            ,q34_data[i],q37_data[i],q42_data[i],q45_data[i],q48_data[i],q51_data[i],q52_data[i],q53_data[i],q54_data[i])
        
        omegabody = omegabody_longlist_lambd[j](
            u1_data[i],u2_data[i],u3_data[i],u4_data[i],u7_data[i],u10_data[i],u13_data[i],u18_data[i],u21_data[i],u24_data[i],u27_data[i],u28_data[i],u31_data[i]
            ,u34_data[i],u37_data[i],u42_data[i],u45_data[i],u48_data[i],u51_data[i],u52_data[i],u53_data[i],u54_data[i]
            ,q1_data[i],q2_data[i],q3_data[i],q4_data[i],q7_data[i],q10_data[i],q13_data[i],q18_data[i],q21_data[i],q24_data[i],q27_data[i],q28_data[i],q31_data[i]
            ,q34_data[i],q37_data[i],q42_data[i],q45_data[i],q48_data[i],q51_data[i],q52_data[i],q53_data[i],q54_data[i])
        
        Inertia_matrix = I_longlist_lambd[j](
            u1_data[i],u2_data[i],u3_data[i],u4_data[i],u7_data[i],u10_data[i],u13_data[i],u18_data[i],u21_data[i],u24_data[i],u27_data[i],u28_data[i],u31_data[i]
            ,u34_data[i],u37_data[i],u42_data[i],u45_data[i],u48_data[i],u51_data[i],u52_data[i],u53_data[i],u54_data[i]
            ,q1_data[i],q2_data[i],q3_data[i],q4_data[i],q7_data[i],q10_data[i],q13_data[i],q18_data[i],q21_data[i],q24_data[i],q27_data[i],q28_data[i],q31_data[i]
            ,q34_data[i],q37_data[i],q42_data[i],q45_data[i],q48_data[i],q51_data[i],q52_data[i],q53_data[i],q54_data[i])
        
        spring_disp = spring_disp_lambd[j](
            u1_data[i],u2_data[i],u3_data[i],u4_data[i],u7_data[i],u10_data[i],u13_data[i],u18_data[i],u21_data[i],u24_data[i],u27_data[i],u28_data[i],u31_data[i]
            ,u34_data[i],u37_data[i],u42_data[i],u45_data[i],u48_data[i],u51_data[i],u52_data[i],u53_data[i],u54_data[i]
            ,q1_data[i],q2_data[i],q3_data[i],q4_data[i],q7_data[i],q10_data[i],q13_data[i],q18_data[i],q21_data[i],q24_data[i],q27_data[i],q28_data[i],q31_data[i]
            ,q34_data[i],q37_data[i],q42_data[i],q45_data[i],q48_data[i],q51_data[i],q52_data[i],q53_data[i],q54_data[i])

        if i == 0:  # Stowed condition
            An_value = 0
            An_rate_value = 0
        else:
            An_value = An_list_lambd[j](
                u1_data[i],u2_data[i],u3_data[i],u4_data[i],u7_data[i],u10_data[i],u13_data[i],u18_data[i],u21_data[i],u24_data[i],u27_data[i],u28_data[i],u31_data[i]
                ,u34_data[i],u37_data[i],u42_data[i],u45_data[i],u48_data[i],u51_data[i],u52_data[i],u53_data[i],u54_data[i]
                ,q1_data[i],q2_data[i],q3_data[i],q4_data[i],q7_data[i],q10_data[i],q13_data[i],q18_data[i],q21_data[i],q24_data[i],q27_data[i],q28_data[i],q31_data[i]
                ,q34_data[i],q37_data[i],q42_data[i],q45_data[i],q48_data[i],q51_data[i],q52_data[i],q53_data[i],q54_data[i]) - \
                An_list_lambd[j](
                u1_data[i-1],u2_data[i-1],u3_data[i-1],u4_data[i-1],u7_data[i-1],u10_data[i-1],u13_data[i-1],u18_data[i-1],u21_data[i-1],u24_data[i-1],u27_data[i-1],u28_data[i-1],u31_data[i-1]
                ,u34_data[i-1],u37_data[i-1],u42_data[i-1],u45_data[i-1],u48_data[i-1],u51_data[i-1],u52_data[i-1],u53_data[i-1],u54_data[i-1]
                ,q1_data[i-1],q2_data[i-1],q3_data[i-1],q4_data[i-1],q7_data[i-1],q10_data[i-1],q13_data[i-1],q18_data[i-1],q21_data[i-1],q24_data[i-1],q27_data[i-1],q28_data[i-1],q31_data[i-1]
                ,q34_data[i-1],q37_data[i-1],q42_data[i-1],q45_data[i-1],q48_data[i-1],q51_data[i-1],q52_data[i-1],q53_data[i-1],q54_data[i-1])

            An_rate_value = An_rate_list_lambd[j](
                u1_data[i],u2_data[i],u3_data[i],u4_data[i],u7_data[i],u10_data[i],u13_data[i],u18_data[i],u21_data[i],u24_data[i],u27_data[i],u28_data[i],u31_data[i]
                ,u34_data[i],u37_data[i],u42_data[i],u45_data[i],u48_data[i],u51_data[i],u52_data[i],u53_data[i],u54_data[i]
                ,q1_data[i],q2_data[i],q3_data[i],q4_data[i],q7_data[i],q10_data[i],q13_data[i],q18_data[i],q21_data[i],q24_data[i],q27_data[i],q28_data[i],q31_data[i]
                ,q34_data[i],q37_data[i],q42_data[i],q45_data[i],q48_data[i],q51_data[i],q52_data[i],q53_data[i],q54_data[i])


        # Computing the system Linear Kinetic Energy
        Ecin_lin[i] = Ecin_lin[i] + 0.5*M_longlist[j]*(np.linalg.norm(vbody))**2

        # Computing the system Angular Kinetic Energy
        Ecin_rot[i] = (Ecin_rot[i] + 0.5*(omegabody.T @ Inertia_matrix @ omegabody)).item()


        # Calculating the system potential energy. In our case, the only potential energy we have is the one stored inside
        # the pre-loaded springs. Therefore
        if j == 0:         
            # Potential Energy of the bus
            Epot[i] = 0 
            # Dissipative forces of the bus
            W_diss_in[i] = 0
        else:
            # Computing the stored potential energy in the springs
            Epot[i] = Epot[i] + 0.5*k_list[j]*spring_disp**2
            
            # Instantaneous Work done by dissipative forces
            W_diss_in[i] = W_diss_in[i] + abs((c_list[j]*An_rate_value)*An_value)
           



    # Computing the system Total Kinetic Energy
    Ecin[i] = Ecin_lin[i] + Ecin_rot[i]

    # Calculating the total mechanical energy at each time step
    Emecc[i] = Ecin[i] + Epot[i]

    if i == 0:
        W_diss_tot[i] = 0
    else:
        # Calculating the total Work done by dissipative forces at each time step
        W_diss_tot[i] = W_diss_tot[i-1] +  W_diss_in[i]

    # Computing the total system energy by summing the system mechanical energy and the dissipated energy
    Etot[i] = Emecc[i] + W_diss_tot[i]

    print(i)



# ---------------------------------------- SOLUTIONS GRAPHS ------------------------------------ #

# Energy_delta = np.max(Etot[700:2000]) - np.min(Etot[700:2000])
# print(f"Maximum energy delta {Energy_delta}")
# print(f"Maximum energy delta percentage {(Energy_delta/np.average(Etot[700:2000]))*100}")



# ------------------------------------------------------------------ #
# Energy conservation

plt.figure(figsize=(10, 6))

plt.subplot(4, 1, 1)
plt.xlim(0,10)
# plt.plot(t_loaded, Ecin_lin, label=r"$K_{\mathrm{lin}}$", color='#456773')
plt.plot(t_loaded, Ecin_lin, label=r"$K_{\mathrm{lin}}$", color='#5C8A99')
plt.xlabel("t [s]")
plt.ylabel("Energy [J]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.xlim(0,10)
# plt.plot(t_loaded, Ecin_rot, label=r"$K_{\mathrm{ang}}$", color='#54793e')
plt.plot(t_loaded, Ecin_rot, label=r"$K_{\mathrm{rot}}$", color='#6A994E')
plt.xlabel("t [s]")
plt.ylabel("Energy [J]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.xlim(0,10)
# plt.plot(t_loaded, Epot, label=r"$U$", color='#d46d0c')
plt.plot(t_loaded, Epot, label=r"$U$", color='#F38B2B')
plt.xlabel("t [s]")
plt.ylabel("Energy [J]")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.xlim(0,10)
# plt.plot(t_loaded, Emecc, label=r"$E_{\mathrm{mecc}}$", color='#7a0c00')
plt.plot(t_loaded, Emecc, label=r"$E_{\mathrm{mecc}}$", color='#BA1200')
plt.xlabel("t [s]")
plt.ylabel("Energy [J]")
plt.grid(True)
plt.legend()


# plt.subplot(5, 1, 5)
# plt.plot(t_loaded, W_diss_tot, label="Energy Dissipated", color='k')
# plt.xlabel("Time [s]")
# plt.ylabel("Energy [Joule]")
# plt.grid(True)
# plt.legend()

plt.tight_layout()


# Grafico unico con tutte le energie
plt.figure(figsize=(10, 6))
plt.xlim(0,10)
plt.plot(t_loaded, Ecin, label=r"$K$", color='#639274')
# plt.plot(t_loaded, Ecin_rot, label="Rotational energy", color='#54793e')
plt.plot(t_loaded, Epot, label=r"$U$", color='#F38B2B')
plt.plot(t_loaded, Emecc, label=r"$E_{\mathrm{mecc}}$", color='#BA1200')
plt.plot(t_loaded, W_diss_tot, label=r"$E_{\mathrm{diss}}$", color='k')
plt.plot(t_loaded, Etot, label="System Total Energy", color='#525252', linestyle = '--' )

plt.xlabel("Time [s]")
plt.ylabel("Energy [J]")
plt.title("System Energy Behavior During Arm Deployment")
plt.grid(True)
plt.legend()
plt.tight_layout()


print('')
print(f'The maximum error in the energy computation is: {np.max(Etot) - np.min(Etot):.4f} J')
print(f'In percentage, that is: {((np.max(Etot) - np.min(Etot)) / np.average(Etot)) * 100:.4f}%')
print('')


plt.show()


print('Codice terminato')
