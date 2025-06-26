# ---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI----------------------------------------------- #
# ------------------------------------------ SCRIPT OBJECTIVE --------------------------------------------------------- #

# The objective of this script is to derive the eq. of motion of a system by applying kane's algorithm.

# The script will be adjusted case by case by the user, following the instruction prompted on the terminal

# Kane's algorithm stars by defining the position vectors of the center of mass of the bodies, and then
# derive them along time to obtain the velocity and acceleration vectors for the two points.
# We then derive the partial velocity vectors from the total velocity vectors of the two center of mass, and then
# we proceed to calculate the generalized forces acting on our system.
# We then proceed to assemble's kane's eq. of motion to obtain, by symplification, the eq of motion of the system.

# This is going to be version 1 of "Kane's method matrix form rearranged"


# ------------------------------------ IMPORTING THE NECESSARY PACKAGES ------------------------------------------------- #

import sympy as sy
import sympy.physics.mechanics as me
import os
import time

#Cleaning the terminal 
os.system('cls')


# ---------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

start = time.perf_counter()


# ---------------------------------------- SYSTEM GENERAL PARAMETERS ------------------------------------------------- #   

nb = 3                     # Number of bodies present 
nj = 2                     # Number of joints presents


# Calculations
n_g = 6 + 3*(nj)        # Numer of gen. coordinates and speeds to define


# For the sake of brevity, we define "Ii" as the identity 3x3 matrix
Ii = sy.eye(3)

# For the sake of brevity, we define "Io" as a zeros 3x3 matrix
Io = sy.zeros(3,3)


# ------------------------------------------ SYMBOLIC VARIABLES ---------------------------------------------------- #    

# Initial motion parameters
t = sy.symbols('t')                # Definition of the time variable
g = sy.symbols('g')                # Gravitational acceleration [m/s^2]
m = sy.symbols('m')                # Pendulum Masses [kg]
k = sy.symbols('k')                # spring constant [N/m]
l = sy.symbols('l')                # Rod length [m]



# ---------------------------------- GENERALIZED COORDINATES AND SPEEDS DEFINITION ------------------------------------- #

# Generalized Coordinates and Speeds definition. These are going to be putted inside two lists, creating the Generalized System
# body state vector and State Vector
BdStVec_list_gen = [me.dynamicsymbols(f'q{i+1}') for i in range(n_g)]
StVec_list_gen = [me.dynamicsymbols(f'u{i+1}') for i in range(n_g)]


# We need also to define them outside of a list, so they can be  used to define ref. systems and forces
for i in range(n_g):
    exec(f'q{i+1} = me.dynamicsymbols(f"q{i+1}")')
    exec(f'u{i+1} = me.dynamicsymbols(f"u{i+1}")')



# Dictionary to subsitute the time derivatives of the gen. coordinates with the gen. speeds
gencoordDer_to_speeds_dict = dict(zip(sy.Matrix([BdStVec_list_gen]).diff(), sy.Matrix([StVec_list_gen]))) 

# Dictionary to subsitute the double time derivatives of the gen. coordinates with the time derivative of the gen. speeds
gencoordDerDer_to_speedsDer_dict = dict(zip(sy.Matrix([BdStVec_list_gen]).diff(t,2), sy.Matrix([StVec_list_gen]).diff(t)))



# ----------------------------------------- REFERENCE FRAME DEFINITION -------------------------------------------------- #


# Inertial Reference Frame definition
n = me.ReferenceFrame('n')          # Inertial ref. Frame, fixed with the pavement   

# Bodies Ref. Frame definitions
b0 = me.ReferenceFrame('b0')        # Root ref. Frames fixed at the pavement, with origin centered on point B0
b1 = me.ReferenceFrame('b1')        # Pendulum body ref. Frame, with origin centered on point B1
b2 = me.ReferenceFrame('b2')        # Pendulum body ref. Frame, with origin centered on point B2

# Inertial Reference Frame origin definition
O = me.Point('O')                   # Inertial ref. Frame origin


# Bodies Ref. Frame origin definition
B0 = me.Point('B0')                 # Root body ref. Frame origin
B1 = me.Point('B1')                 # First Pendulum body ref. Frame origin
B2 = me.Point('B2')                 # Second Pendulum body ref. Frame origin

# Joints points definitions
G0 = me.Point('G0')                 # Joint G0
G1 = me.Point('G1')                 # Joint G1



# Setting the relative position between the frame's origins
B0.set_pos(O, 0*n.x + 0*n.y + 0*n.z)                         # Setting point B0 relative to the inertial ref. Frame
B0.set_pos(B0, 0*b0.x + 0*b0.y + 0*b0.z)                     # Setting point B0 as the origin of the root ref. frame

B1.set_pos(B0, (l/2)*b1.x + 0*b1.y + 0*b1.z)                 # Setting the point B1 relative to the joint position
B1.set_pos(B1, 0*b1.x + 0*b1.y + 0*b1.z )                    # Setting the point B1 as the origin of the pendulum ref. frame

B2.set_pos(B1, (l/2)*b1.x + 0*b1.y + 0*b1.z)                 # Setting the point B1 relative to the joint position
B2.set_pos(B2, 0*b2.x + 0*b2.y + 0*b2.z )                    # Setting the point B1 as the origin of the pendulum ref. frame


# Setting the position of the joint
G0.set_pos(B0, 0*b0.x + 0*b0.y + 0*b0.z)                     # Setting point B0 as the origin of the root ref. frame
G1.set_pos(B1, (l/2)*b1.x + 0*b1.y + 0*b1.z)                  # Setting point B0 as the origin of the root ref. frame


# --------------------------------------------------------------------------- #
# Setting the relative orientation between the frames

# Defining the ficticious ref. frame as rotated wrt the Inertial
b0.orient_body_fixed(n, (0, 0, 0), 'zxy')

# Defining the pendulum ref. frame as rotated wrt to the root body ref. frame                                
b1.orient_body_fixed(b0, (q6, 0, 0), 'zxy')

# Defining the pendulum ref. frame as rotated of 0° wrt to the joint ref. frame                                
b2.orient_body_fixed(b1, (0, q7, 0), 'zxy')


# Setting the velocity of the inertial ref. frame to 0, to allow the calculations to be developed
O.set_vel(n, 0)
n.set_ang_vel(n, 0)


# Putting the body ref. Frame inside a list, so i can express vector quantities easier
ref = [b0, b1, b2]



# ----------------------------------------------- JOINTS GEN. SPEEDS ------------------------------------------------- #   

# Here we need to define the generalized speeds vector for each joint. That is, the velocity of how each joint turns 
# wrt the "inner body" its attacched to. 

# We then need to build a vector containing all the generalized speeds of the joints
omegajoint_list = StVec_list_gen[3:(n_g - 3)]



# ------------------------------------------------- JOINTS PARTIALS -------------------------------------------------- #

# Here we need to calculate "the joints partials". These are rotational matrices that "tells" how the joint rotation around
# its axis are coupled between eachother

GAMMA_g0 = sy.Matrix([[1,0,0],
                      [0,1,0],
                      [0,0,1]
                      ])

GAMMA_g1 = sy.Matrix([[1,0,0],
                      [0,1,0],
                      [0,0,1]
                      ])


# Here we need to build a list, that enclose all the Joint partials we calculated before. In this way we can use this list
# to easily use the joint partials for our calculations
GAMMA_List = [GAMMA_g0, GAMMA_g1]     



# ---------------------------------------------- DIRECTION COSINE MATRICES -------------------------------------------- #

# Here we need to calculate "direction cosine matrix". 

# To compute the OMEGA matrix, we need to calculate the direction cosine matrices, which describe the relative orientation of
# the bodies in the system with respect to each other
Cb0_b1 = b1.dcm(b0)
Cb0_b2 = b2.dcm(b0)
Cb1_b2 = b2.dcm(b1)

# ------------------------------------------------------ #

# For the bigV matrix, we need to calculate the direction cosine matrices that "tells" how to bodies are oriented wrt
# the inertial ref frame
Cb0_n = n.dcm(b0)
Cb1_n = n.dcm(b1)
Cb2_n = n.dcm(b2)



# ---------------------------------------------VECTORS CALCULATIONS-------------------------------------------------- #

# Here we are going to define the distance vectors between the bodies center of mass and the joints
rG0_B0 = G0.pos_from(B0)
rG0_B1 = G0.pos_from(B1)
rG0_B2 = G0.pos_from(B2)
rG1_B1 = G1.pos_from(B1)
rG1_B2 = G1.pos_from(B2)


# For the residual accelerations we need to calculate the position vectors difference between the bodies cof and shared joints 
# For now, we can put vectors inside a list to automatically calculate the residual accelerations, but this works ONLY if
# the system have only "one branch" connecting all the bodies.
r_i_o = [rG0_B0, rG0_B1, rG1_B1, rG1_B2]


# Here we are going to define the "beta" coefficient, thats the vector connecting the mass center of body Bi with the root body
beta1 = rG0_B1-rG0_B0
beta2 = rG1_B2-rG1_B1 + rG0_B1-rG0_B0


# For the residual accelerations we need to calculate the angular velocities that refers to inner and outer bodies wrt
# to the shared joints. For now, we can put these velocities inside a list to automatically calculate the residual accelerations,
# but this works ONLY if the system have only "one branch" connecting all the bodies.
omegabody_list = [b0.ang_vel_in(n), b1.ang_vel_in(n), b2.ang_vel_in(n)]



# -------------------------------------------- PARTIAL VELOCITIES MATRICES ------------------------------------------------ #

# Here we will have to manually compile the Angular and linear velocities matrices, by inspecting how the bodies are connected 
# between each other. These matrices will be (n+1) x (n), where n = n° of bodies


# Angular partial velocity Matrix
OMEGA = sy.BlockMatrix([[Ii, Io, Io, Io],
                        [Cb0_b1, GAMMA_g0, Io, Io],
                        [Cb0_b2, Cb1_b2*GAMMA_g0, GAMMA_g1, Io]
                        ])


# Linear partial velocity Matrix
bigV = sy.BlockMatrix([[Io, Io, Io, Ii],
                       [beta1.to_matrix(n).hat()*Cb0_n, rG0_B1.to_matrix(n).hat()*Cb1_n*GAMMA_g0 , Io, Ii],
                       [beta2.to_matrix(n).hat()*Cb0_n, rG0_B2.to_matrix(n).hat()*Cb1_n*GAMMA_g0, rG1_B2.to_matrix(n).hat()*Cb2_n*GAMMA_g1, Ii]
                       ])



# ------------------------------------------- REMAINDER ACCELERATIONS ----------------------------------------------------- #

# Here we are going to calculate the remainder accelerations vectors for the bodies, both linear and angular.
# We can do it recursively, starting from the "root body" and going to "leaf bodies"

# The remainder accs. of the root body are set to zero as default
a_rem0 = sy.Matrix([[0],[0],[0]])
alfa_rem0 = sy.Matrix([[0],[0],[0]])

# Linear Acc. Remainder vector list
a_rem = [0]*nb
a_rem[0] = a_rem0    

# Angular Acc. Remainder vector list
alfa_rem = [0]*nb
alfa_rem[0] = alfa_rem0 


#---------------------------------------------------#

# Then we are going to calculate the remainder accelerations
for i in range(nj):
   
   alfa_rem[i+1] =  alfa_rem[i] + ((GAMMA_List[i].diff(t)*sy.Matrix(omegajoint_list[(3*i):(3+3*i)])) + omegabody_list[i+1].to_matrix(ref[i+1]).hat()*(GAMMA_List[i]*sy.Matrix(omegajoint_list[(3*i):(3+3*i)])))
   a_rem[i+1] = a_rem[i] + omegabody_list[i].to_matrix(n).hat()*(omegabody_list[i].to_matrix(n).hat()*(r_i_o[2*i].to_matrix(n))) + sy.Matrix(alfa_rem[i]).hat()*(r_i_o[2*i].to_matrix(n)) - omegabody_list[i+1].to_matrix(n).hat()*(omegabody_list[i+1].to_matrix(n).hat()*(r_i_o[2*i+1].to_matrix(n))) - sy.Matrix(alfa_rem[i+1]).hat()*(r_i_o[2*i+1].to_matrix(n))
   
   # Adding an if statement to exit early the for cycle 
   if i == nj-1:
      break 
   


# ----------------------------------------- FORCES & TORQUES ACTING ON OUR BODIES---------------------------------------------- #

# For now, specifying the point in which the force vector is applied doesnt change anything. I hope in the future to make the code able to derive the equivalent
# force system applied to the c.o.m of the bodies, specifying the forces on the position in which are really applied.

# Forces acting on point B0
f_01 = (0)*b0.x
F_01 = (B0, f_01)


# Forces acting on point B1, com of the first pendulum
f_11 = (m*g)*n.x
F_11 = (B1, f_11)


# Forces acting on point B2, com of the second pendulum
f_21 = (m*g)*n.x
F_21 = (B2, f_21)


#---------------------------------------------------#

# Torques acting on the B0
t_01 = 0*n.z
T_01 = (B0, t_01)


# Torques acting on the first pendulum B1
t_11 = (-k*q6)*n.z + (k*q7)*b1.x
T_11 = (B1, t_11)


# Torques acting on the second pendulum B2
t_21 = (-k*q7)*b1.x
T_21 = (B2, t_21)


#---------------------------------------------------#

# Calculating the resultant forces & torques acting on the two bodies
Ftot_0 = F_01[1]                             #Total force vector acting on the cart
Ttot_0 = T_01[1]                             #Total torque vector acting on the cart

Ftot_1 = F_11[1]                             #Total force vector acting on the first pendulum
Ttot_1 = T_11[1]                             #Total torque vector acting on the first pendulum

Ftot_2 = F_21[1]                             #Total force vector acting on the second pendulum
Ttot_2 = T_21[1]                             #Total torque vector acting on the second pendulum


#---------------------------------------------------#

# Writing the vectors as list for easy of use
F_list = [Ftot_0.to_matrix(n), Ftot_1.to_matrix(n), Ftot_2.to_matrix(n)]
T_list = [Ttot_0.to_matrix(b0), Ttot_1.to_matrix(b1), Ttot_2.to_matrix(b2)]



# -------------------------------------------- BODIES MASSES AND INERTIAS ------------------------------------------------- #

# Here we have to define the masses and inertia matrices of each body present in our system.

# Since we defined a body ref. frame centered in each body com and alligned with its principal inertia axis, we can
# easily define the inertia matrix for each body using the outer product:

# Mass List
M_list = [0, m, m]


# Inertia Dyadic for each body, centered in their respective com and expressed along their principal inertia axis
I_0 = (0)*m*(l**2)*(me.outer(b0.x, b0.x) + me.outer(b0.y, b0.y) + me.outer(b0.z, b0.z))
I_1 = (1/12)*m*(l**2)*(0*me.outer(b1.x, b1.x) + me.outer(b1.y, b1.y) + me.outer(b1.z, b1.z))
I_2 = (1/12)*m*(l**2)*(me.outer(b2.x, b2.x) + 0*me.outer(b2.y, b2.y) + me.outer(b2.z, b2.z))


# Inertia List
I_list = [I_0, I_1, I_2]



# ---------------------------------------------- BUILDING MATRIX FORM --------------------------------------------------- #

# With all the building blocks ready, its time to assemble the two big matrices that will enable us to solve the motion of
# the system: COEF e RHS matrices
# But first, we need to derive the matrices from the lists we wrote before


# Mass Matrix 
M = sy.BlockMatrix([[Ii*M_list[0], Io, Io], 
                    [Io, Ii*M_list[1], Io],
                    [Io, Io, Ii*M_list[2]]
                    ]) 

# Inertia Matrix
I = sy.BlockMatrix([[I_list[0].to_matrix(b0), Io, Io],
                    [Io, I_list[1].to_matrix(b1), Io],
                    [Io, Io, I_list[2].to_matrix(b2)]
                    ])


# COEF Matrix
COEF = sy.Matrix(OMEGA.T)*sy.Matrix(I)*sy.Matrix(OMEGA) + sy.Matrix(bigV.T)*sy.Matrix(M)*sy.Matrix(bigV)


# Calculating the vector product between the angular velocities and Angular momentum of each body
wxH = [0]*nb

for i in range(nb):
        wxH[i] = me.cross(omegabody_list[i], I_list[i]).dot(omegabody_list[i]).to_matrix(ref[i])   
           
# RHS Matrix
RHS = sy.Matrix(OMEGA.T)*(sy.Matrix(T_list) - sy.Matrix(I)*sy.Matrix(alfa_rem) - sy.Matrix(wxH)) + sy.Matrix(bigV.T)*(sy.Matrix(F_list) - sy.Matrix(M)*sy.Matrix(a_rem))


# -------------------------------------------------- BLOCKING DOF ---------------------------------------------------- #

# Here we want to "block" the dof of the system. Right now the problem should be defined in the most general way possible
# In case there are dof blocked, like the motion is planar... we should "tell" the code right now. In this way, we can 
# safe some computational time, istead of letting the code run with the most general version of the problem and turn off 
# unnecessary dof at the end 

# We will use a dictionary that will set to zero each gen. coordinates/speeds, blocking that dof for the system. 
gencoord_active = sy.Matrix([0, 0, 0, 0, 0, q6, q7, 0, 0, 0, 0, 0])
genspeeds_active = sy.Matrix([0, 0, 0, 0, 0, u6, u7, 0, 0, 0, 0, 0])

# Dictionary to turn off the dof of the system
gencoord_dict = dict(zip(BdStVec_list_gen ,gencoord_active))
genspeeds_dict = dict(zip(StVec_list_gen ,genspeeds_active))



# ---------------------------------------------------- ADJUSTING UP ------------------------------------------------------ #

# Now we turn off the system dof we dont need by using the dictionaries we initially created
COEF = COEF.xreplace(gencoordDerDer_to_speedsDer_dict).xreplace(gencoordDer_to_speeds_dict).xreplace(genspeeds_dict).xreplace(gencoord_dict)
RHS = RHS.xreplace(gencoordDerDer_to_speedsDer_dict).xreplace(gencoordDer_to_speeds_dict).xreplace(genspeeds_dict).xreplace(gencoord_dict) 


# In the same way, we set to 0 the Body state vector and vector values associated with the blocked dof
BdStVec_list_gen = [expr.xreplace(gencoord_dict) for expr in BdStVec_list_gen]
StVec_list_gen = [expr.xreplace(genspeeds_dict) for expr in StVec_list_gen]


# Next, we are going to eliminate the rows and columns of the COEF matrix and RHS vector associated with the blocked dof

# Search for the index values associated with the blocked dof
index_to_erase = [i for i, value in enumerate(gencoord_active) if value == 0]

# Erasing the corresponding rows
for index in reversed(index_to_erase):  # Reverse order to avoid altering indexes
    COEF.row_del(index)
    RHS.row_del(index)  
 

# Erasing the corresponding coloumns
for index in reversed(index_to_erase):  # Reverse order to avoid altering indexes
    COEF.col_del(index)



# At last, we need to create the State vector(u) and body state vector(x) for the system
BdStVec = sy.Matrix([term for term in gencoord_active if term != 0])
StVec = sy.Matrix([term for term in genspeeds_active if term != 0])



# ------------------------------------------------ PRINTING THE MATRICES ----------------------------------------------- #

# Printing the results 
sy.pprint(sy.simplify(COEF))
print()
print()
sy.pprint(sy.simplify(RHS))
print()
print()
sy.pprint(sy.simplify(StVec.diff()))
print()
print()


# Recovering the Motion Equations for the system
sy.pprint(sy.simplify(COEF*StVec.diff()-RHS))




# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')














