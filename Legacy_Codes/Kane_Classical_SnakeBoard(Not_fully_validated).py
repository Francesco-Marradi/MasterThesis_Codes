#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI-----------------------------------------------#
#---------------------------------------------SCRIPT OBJECTIVE---------------------------------------------------------#

# The objective of this script is to derive the eq. of motion of a system by applying kane's algorithm.

# The script will be adjusted case by case by the user, following the instruction prompted on the terminal

# Kane's algorithm stars by defining the position vectors of the center of mass of the bodies, and then
# derive them along time to obtain the velocity and acceleration vectors for the two points.
# We then derive the partial velocity vectors from the total velocity vectors of the two center of mass, and then
# we proceed to calculate the generalized forces acting on our system.
# We then proceed to assemble's kane's eq. of motion to obtain, by symplification, the eq of motion of the system.




# ------------------------------------ IMPORTING THE NECESSARY PACKAGES ------------------------------------------------- #

import numpy as np
import sympy as sy
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import time

#Cleaning the terminal 
os.system('cls')


# ---------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

start = time.perf_counter()


# ---------------------------------------- SYSTEM GENERAL PARAMETERS ------------------------------------------------- #

n = 3                      # Number of bodies present 
m_h = 15                  # Number of constraints present 
m_nh = 2                   # Number of Non-Holonomic constrains present


# Calculations
dof = 6*n - m_h            # System's Degree of freedom, tells us the n° of Indipendent gen. Coordinates we will have
n_coor = dof + m_nh        # Number of gen. coordinates we have to define


# Generalized Coordinates definition (n° equal to n_coor)
q1 = me.dynamicsymbols('q1')
q2 = me.dynamicsymbols('q2')
q3 = me.dynamicsymbols('q3')
q4 = me.dynamicsymbols('q4')
q5 = me.dynamicsymbols('q5')


# Generalized Speeds definition (n° equal to n_coor)
u1 = me.dynamicsymbols('u1')      
u2 = me.dynamicsymbols('u2')      
u3 = me.dynamicsymbols('u3')      
u4 = me.dynamicsymbols('u4')
u5 = me.dynamicsymbols('u5')


# -------------------------------------------- SYMBOLIC VARIABLES ---------------------------------------------------- #

# Initial motion parameters
t = sy.symbols('t')                # Definition of the time variable
g = sy.symbols('g')                # Gravitational acceleration [m/s^2]
m1 = sy.symbols('m1')              # Mass of body 1 [kg]
m2 = sy.symbols('m2')              # Mass of body 2 [kg]
b = sy.symbols('b')                # Damper coefficient [N*s/m]
k = sy.symbols('k')                # spring constant [N/m]
l = sy.symbols('l')                # Rod length [m]
I = sy.symbols('I')                # Inertia [kg*m^2]


# # Inertia of the cart, along z axis [kg*m^2].
# I_m1 = sy.Matrix([[0,0,0],
#                   [0,(1/12)*m1*(l**2),0],
#                   [0,0,(1/12)*m1*(l**2)]])     

# # Inertia of the pendulum, along z axis [kg*m^2]. 
# I_m2 = sy.Matrix([[(1/12)*m1*(l**2),0,0],
#                   [0,0,0],
#                   [0,0,(1/12)*m1*(l**2)]])           




# ----------------------------------------- REFERENCE FRAME DEFINITION -------------------------------------------------- #

# Reference Frame definition
N = me.ReferenceFrame('N')          # Inertial ref. Frame, fixed with the pavement   
A = me.ReferenceFrame('A')        # Cart body ref. Frame, with origin centered on point A
B = me.ReferenceFrame('B')        # Pendulum body ref. Frame, with origin centered on  point G
C = me.ReferenceFrame('C')        # Pendulum body ref. Frame, with origin centered on  point G


# Reference Frame's origin definition
O = me.Point('O')                   # Inertial ref. Frame origin
A0 = me.Point('Ao')                 # Cart ref. Frame origin
B0 = me.Point('Bo')                 # Pendulum body ref. Frame origin
C0 = me.Point('Co')                 # Pendulum body ref. Frame origin

# Setting the relative position between the frame's origins
A0.set_pos(O, q1*N.x + q2*N.y)                                 # Setting point A relative to the inertial ref. Frame
A0.set_pos(A0, 0*A.x + 0*A.y + 0*A.z)                           # Setting point A as the origin of the cart ref. frame

B0.set_pos(A0, (l/2)*A.x)                                      # Setting point A relative to the cart ref. Frame
B0.set_pos(B0, 0*B.x + 0*B.y + 0*B.z)                          # Setting point G as the origin of the prendulum ref. frame

C0.set_pos(A0, -(l/2)*A.x)                                      # Setting point A relative to the cart ref. Frame
C0.set_pos(C0, 0*C.x + 0*C.y + 0*C.z)                          # Setting point G as the origin of the prendulum ref. frame


# Setting the relative orientation between the frames
A.orient_axis(N, q3, N.z)                                     #Defining the cart ref. frame as rotated of 0° wrt the Inertial. By doing so, i tell the code that the two frames have their axes alligned
B.orient_axis(A, -(sy.pi/2 - q4), A.z)                         #Defining the pendulum ref. frame as rotated of q2 wrt to the cart ref. frame
C.orient_axis(A, -(sy.pi/2 - q5), A.z) 


# Setting the velocity of the inertial ref. frame to 0, to allow the calculations to be developed
O.set_vel(N, 0)
N.set_ang_vel(N, 0)



# Bodies Inertia written in their body ref. frame
I_A_Ao = I_A_Ao = I*me.outer(A.z, A.z)
I_B_Bo = I_B_Bo = I*me.outer(B.z, B.z)
I_C_Co = I*me.outer(C.z, C.z)

# Bodies Inertia transformed to the inertia matrix wrt inertial ref. system N 
I_m1 =  I_A_Ao.to_matrix(N)
I_m2 =  I_B_Bo.to_matrix(N)
I_m3 =  I_C_Co.to_matrix(N)    



# ---------------------------------------------- VECTORS CALCULATIONS -------------------------------------------------- #

# Position vector definition
x_A = A0.pos_from(O)
x_B = B0.pos_from(O)
x_C = C0.pos_from(O)


# Velocity vector definition
v_A = A0.vel(N)
v_B = B0.vel(N)
v_C = C0.vel(N)

# Acceleration vector definition
a_A = A0.acc(N)
a_B = B0.acc(N)
a_C = C0.acc(N)

# Angular velocity vector
omega_A = A.ang_vel_in(N)
omega_B = B.ang_vel_in(N)
omega_C = C.ang_vel_in(N)

# Angular acceleration vector definition
omegadot_A = A.ang_acc_in(N) 
omegadot_B = B.ang_acc_in(N)
omegadot_C = C.ang_acc_in(N)



# ------------------------------------------- GENERALIZED SPEEDS ----------------------------------------------------- #

# Projected Angular velocities 
omega_i = [omega_A.dot(A.x), omega_A.dot(A.y), omega_A.dot(A.z), 
           omega_B.dot(B.x), omega_B.dot(B.y), omega_B.dot(B.z),
           omega_C.dot(C.x), omega_C.dot(C.y), omega_C.dot(C.z),]

sy.pprint(omega_i)


# Dictionary with the substitution to do  #FORSE DA SPOSTARE A DOPO I CONSTRAINTS HOLONOMICI E NON HOLO
substitutions = {
     
    q1.diff(): u1, 
    q2.diff(): u2,
    q3.diff(): u3, 
    q4.diff(): u4,
    q5.diff(): u5,
    q1.diff(t,2): u1.diff(),
    q2.diff(t,2): u2.diff(),
    q3.diff(t,2): u3.diff(),
    q4.diff(t,2): u4.diff(),
    q5.diff(t,2): u5.diff(),

    }     


# Vectors rewriting
v_A = v_A.xreplace(substitutions)
v_B = v_B.xreplace(substitutions)
v_C = v_C.xreplace(substitutions)
a_A = a_A.xreplace(substitutions)
a_B = a_B.xreplace(substitutions)
a_C = a_C.xreplace(substitutions)
omega_A = omega_A.xreplace(substitutions)
omega_B = omega_B.xreplace(substitutions)
omega_C = omega_C.xreplace(substitutions)
omegadot_A = omegadot_A.xreplace(substitutions)
omegadot_B = omegadot_B.xreplace(substitutions)
omegadot_C = omegadot_C.xreplace(substitutions)




# --------------------------------------------- NON-HOLONOMIC CONSTRAINTS ----------------------------------------------------- #

# Non Holonomic constraints definition
# Here we need to define all  non-holonomic constraint and the project it along the axis of a ref. system to obtain obtain the constraints equations. 
# "fn" will be the vector containing this constraints equations.
fn = sy.Matrix([B0.vel(N).dot(B.x), C0.vel(N).dot(C.x)])

# Rewrite it in terms of the generalized speeds 
fn = fn.xreplace(substitutions)


# ----------------------------------------------- HOLONOMIC CONSTRAINTS ------------------------------------------------------- #

# Holonomic constraints definition
# Here we need to define all holonomic constraint and the project it along the axis of a ref. system to obtain the constraints equations. 
# "fh" will be the vector containing this constraints equations.
fh = sy.Matrix([(0*N.x).dot(N.x), (0*N.x).dot(N.x)])

# Rewrite it in terms of the generalized speeds 
fh = fh.xreplace(substitutions)


# ------------------------------------------------- MOTION CONSTRAINTS -------------------------------------------------------- #

# Motion constraints derivation
# Non-holonomic constraints are already in the correct form (of motion constraints) so they dont need to be derived. 

# Here we have to differentiate the holonomic constraints with respect to time to arrive at a motion constraint. We then substitute 
# the all the derivatives of the coordinates with their respective speeds.
fhd = fh.diff(t).xreplace(substitutions)


# Next step is to choose the dependent speeds from the motion equations, we can do that by inspection of the eqs. themselves
# sy.pprint(fhd)
# print()
# print()
# sy.pprint(fn)



# -------------------------------- INDEPENDENT AND DEPENDENT GEN. COORDINATES AND SPEEDS ---------------------------------------- #

# We now define the u vector and its derivative, containing the all the independent gen. speeds we choosed before
u = sy.Matrix([u3, u4, u5])
ud = u.diff(t)


# Now define the ur vector and its derivative, containing the all the dependent gen. speeds we choosed before
ur = sy.Matrix([u1, u2])
urd = ur.diff(t)


# And at last, we assemble the qN vector containing all the gen. speeds we have defined
uN = u.col_join(ur)


# Consequently we define the two vectors containing the indepented and dependent gen. coordinates
# Independent Gen.Coordinates Vector
q = sy.Matrix([q3, q4, q5])

# Dependent Gen.Coordinates Vector
qr = sy.Matrix([q1, q2])

# And at last, we assemble the qN vector containing all the gen. coordinates we have defined
qN = q.col_join(qr)


# Next, we need to solve the holonomic linear motion constraints equations for these dependent speeds
# Mhd = fhd.jacobian(ur)
# ur_zero = {ui: 0 for ui in ur}
# ghd = fhd.xreplace(ur_zero)
# ur_solh = Mhd.LUsolve(-ghd)
# ur_replh = dict(zip(u, ur_solh))

# Next, we need to solve the nonholonomic linear motion constraints equations for these dependent speeds
Mn = fn.jacobian(ur)
ur_zero = {ui: 0 for ui in ur}
gn = fn.xreplace(ur_zero)
ur_soln = Mn.LUsolve(-gn)
ur_repln = dict(zip(ur, ur_soln))



# ----------------------------------------------- VECTORS RE-WRITING ------------------------------------------------------- #

# Now we need to rewrite all the velocities and accelerations vectors in terms of independent speeds
# While holonomic constraints could contain both dependet generalized coordinates and speeds, non-holonomic one could also 
# contain dependet gen. accelerations (time derivatives of gen.speeds). Therefore, we need to differentiate the constraints 
# with respect to time and then substituting for the dependent generalized speeds gives us equations for the dependent
# generalized accelerations.


# Velocity vectors rewriting
v_A = v_A.xreplace(ur_repln)
v_B = v_B.xreplace(ur_repln)
v_C = v_C.xreplace(ur_repln)
omega_A = omega_A.xreplace(ur_repln)
omega_B = omega_B.xreplace(ur_repln)
omega_C = omega_C.xreplace(ur_repln)


# Acceleration vectors rewriting
# First, time differentiate the nonholonomic constraints
fnd = fn.diff(t).xreplace(substitutions)

# Now solve for the dependent generalized accelerations
Mnd = fnd.jacobian(urd)
urd_zero = {udi: 0 for udi in urd}
gnd = fnd.xreplace(urd_zero).xreplace(ur_repln)
urd_sol = Mnd.LUsolve(-gnd)
urd_repl = dict(zip(urd, urd_sol))

# At last, we rewrite the acceleration vectors
a_A = a_A.xreplace(ur_repln).xreplace(urd_repl)
a_B = a_B.xreplace(ur_repln).xreplace(urd_repl)
a_C = a_C.xreplace(ur_repln).xreplace(urd_repl)
omegadot_A = omegadot_A.xreplace(ur_repln).xreplace(urd_repl)
omegadot_B = omegadot_B.xreplace(ur_repln).xreplace(urd_repl)
omegadot_C = omegadot_C.xreplace(ur_repln).xreplace(urd_repl)



# ------------------------------------------ PARTIAL VELOCITIES CALCULATIONS--------------------------------------------------- #

# First, we define a vector containing all the velocities we have to work with 
vels = (v_A, v_B, v_C, omega_A, omega_B, omega_C)

# Then, we calculate the partial Velocities for the system using sympy :D
v_A_part, v_B_part, v_C_part, omega_A_part, omega_B_part, omega_C_part = me.partial_velocity(vels, u, N)


# Rearranging the partial velocities into matrices for ease of use
v_A_partM = sy.Matrix([[v_A_part[0].dot(N.x), v_A_part[0].dot(N.y), v_A_part[0].dot(N.z)],
                       [v_A_part[1].dot(N.x), v_A_part[1].dot(N.y), v_A_part[1].dot(N.z)],
                       [v_A_part[2].dot(N.x), v_A_part[2].dot(N.y), v_A_part[2].dot(N.z)]])

v_B_partM = sy.Matrix([[v_B_part[0].dot(N.x), v_B_part[0].dot(N.y), v_B_part[0].dot(N.z)],
                       [v_B_part[1].dot(N.x), v_B_part[1].dot(N.y), v_B_part[1].dot(N.z)],
                       [v_B_part[2].dot(N.x), v_B_part[2].dot(N.y), v_B_part[2].dot(N.z)]])

v_C_partM = sy.Matrix([[v_C_part[0].dot(N.x), v_C_part[0].dot(N.y), v_C_part[0].dot(N.z)],
                       [v_C_part[1].dot(N.x), v_C_part[1].dot(N.y), v_C_part[1].dot(N.z)],
                       [v_C_part[2].dot(N.x), v_C_part[2].dot(N.y), v_C_part[2].dot(N.z)]])

omega_A_partM = sy.Matrix([[omega_A_part[0].dot(N.x), omega_A_part[0].dot(N.y), omega_A_part[0].dot(N.z)],
                           [omega_A_part[1].dot(N.x), omega_A_part[1].dot(N.y), omega_A_part[1].dot(N.z)],
                           [omega_A_part[2].dot(N.x), omega_A_part[2].dot(N.y), omega_A_part[2].dot(N.z)]])

omega_B_partM = sy.Matrix([[omega_B_part[0].dot(N.x), omega_B_part[0].dot(N.y), omega_B_part[0].dot(N.z)],
                           [omega_B_part[1].dot(N.x), omega_B_part[1].dot(N.y), omega_B_part[1].dot(N.z)],
                           [omega_B_part[2].dot(N.x), omega_B_part[2].dot(N.y), omega_B_part[2].dot(N.z)]])

omega_C_partM = sy.Matrix([[omega_C_part[0].dot(N.x), omega_C_part[0].dot(N.y), omega_C_part[0].dot(N.z)],
                           [omega_C_part[1].dot(N.x), omega_C_part[1].dot(N.y), omega_C_part[1].dot(N.z)],
                           [omega_C_part[2].dot(N.x), omega_C_part[2].dot(N.y), omega_C_part[2].dot(N.z)]])



# ----------------------------------------- FORCES & TORQUES ACTING ON OUR BODIES---------------------------------------------- #

# For now, specifying the point in which the force vector is applied doesnt change anything. I hope in the future to make the code able to derive the equivalent
# force system applied to the c.o.m of the bodies, specifying the forces on the position in which are applied.

# Forces acting on point A, com of the cart
f_1 = (0)*N.x


F_1 = (A, f_1)



# Forces acting on point G, com of the pendulum
f_2 = (0)*N.x
F_2 = (B, f_2)


# Writing the vectors as matrices for easy of use
F_1_m = sy.Matrix([F_1[1].dot(N.x), F_1[1].dot(N.y), F_1[1].dot(N.z)])
F_2_m = sy.Matrix([F_2[1].dot(N.x), F_2[1].dot(N.y), F_2[1].dot(N.z)])



# Torques acting on the cart C
t_1 = -0*N.z
t_2 = 0*N.x

T_1 = (A, t_1)
T_2 = (A, t_2)


# Torques acting on the pendulum P
t_3 = 0*N.x
T_3 = (B, t_3)


# Writing the vectors are matrices for easy of use
T_1_m = sy.Matrix([T_1[1].dot(N.x), T_1[1].dot(N.y), T_1[1].dot(N.z)])
T_2_m = sy.Matrix([T_2[1].dot(N.x), T_2[1].dot(N.y), T_2[1].dot(N.z)])
T_3_m = sy.Matrix([T_3[1].dot(N.x), T_3[1].dot(N.y), T_3[1].dot(N.z)])


# Calculating the resultant forces & torques acting on the two bodies
Ftot_1 = F_1_m                                         #Total force vector acting on the cart
Torqtot_1 = T_1_m + T_2_m                          #Total torque vector acting on the cart

Ftot_2 = F_2_m                               #Total force vector acting on the pendulum
Torqtot_2 = T_3_m                            #Total torque vector acting on the pendulum

Ftot_3 = F_2_m                               #Total force vector acting on the pendulum
Torqtot_3 = T_3_m                            #Total torque vector acting on the pendulum



# ---------------------------------------- GENERALIZED INERTIA & ACTIVE FORCES----------------------------------------- #

# Rearranging quantities into matrices for ease of use

# Angular velocity matrix for the entire system
omega = sy.Matrix([[omega_A.dot(N.x), omega_A.dot(N.y), omega_A.dot(N.z)],
                   [omega_B.dot(N.x), omega_B.dot(N.y), omega_B.dot(N.z)],
                   [omega_C.dot(N.x), omega_C.dot(N.y), omega_C.dot(N.z)]])    

# Acceleration matrix for the entire system
a = sy.Matrix([[a_A.dot(N.x), a_A.dot(N.y), a_A.dot(N.z)],
               [a_B.dot(N.x), a_B.dot(N.y), a_B.dot(N.z)],
               [a_C.dot(N.x), a_C.dot(N.y), a_C.dot(N.z)]])    


# Angular acceleration matrix for the entire system
omegadot = sy.Matrix([[omegadot_A.dot(N.x), omegadot_A.dot(N.y), omegadot_A.dot(N.z)],
                      [omegadot_B.dot(N.x), omegadot_B.dot(N.y), omegadot_B.dot(N.z)],
                      [omegadot_C.dot(N.x), omegadot_C.dot(N.y), omegadot_C.dot(N.z)]])

# Mass vector for the entire system
Md = sy.Matrix([m1, m1, m1])

# Force Matrix
Ftot = sy.Matrix([[Ftot_1.T],
                  [Ftot_2.T],
                  [Ftot_3.T],])    

# Torque Matrix
Torqtot =  sy.Matrix([[Torqtot_1.T],
                      [Torqtot_2.T],
                      [Torqtot_3.T],]) 



# Initializing the empty vectors
F = sy.Matrix.zeros(dof,1)         # Generalized inertia forces vertical vector
f = sy.Matrix.zeros(dof,1)         # Generalized forces vertical vector
K = sy.Matrix.zeros(dof,1)         # Kane's eq. vector vertical vector


# Calculating the Generalized forces with a for cycle
for i in range(dof):

   # Generalized Inertia forces
   F[i,0] = - (Md[0] * a[0,0:3].dot(v_A_partM[i,0:3]) + ((omegadot[0,0:3]*I_m1) + omega[0,0:3].cross(I_m1*(omega[0,0:3].T))).dot(omega_A_partM[i,0:3])) - (Md[1] * a[1,0:3].dot(v_B_partM[i,0:3]) + ((omegadot[1,0:3]*I_m2) + omega[1,0:3].cross(I_m2*(omega[1,0:3].T))).dot(omega_B_partM[i,0:3])) - (Md[2] * a[2,0:3].dot(v_B_partM[i,0:3]) + ((omegadot[2,0:3]*I_m2) + omega[2,0:3].cross(I_m2*(omega[2,0:3].T))).dot(omega_B_partM[i,0:3]))

   # Generalized Active forces
   f[i,0] = (Ftot[0,0:3].dot(v_A_partM[i,0:3]) + (Torqtot[0,0:3]).dot(omega_A_partM[i,0:3])) + (Ftot[1,0:3].dot(v_B_partM[i,0:3]) + (Torqtot[1,0:3]).dot(omega_B_partM[i,0:3])) + (Ftot[2,0:3].dot(v_B_partM[i,0:3]) + (Torqtot[2,0:3]).dot(omega_B_partM[i,0:3]))



# ------------------------------------------- ASSEMBLY KANE'S EQ. OF MOTION------------------------------------------- #

# Assembly Kane's Eq. of Motion
for i in range(dof):
   
   K[i,0] = F[i,0] + f[i,0]



# Printing on screen the eq. of motion derived
sy.pprint(sy.simplify(K[0]))
print()
print()
sy.pprint(sy.simplify(K[1]))
print()
print()
sy.pprint(sy.simplify(K[2]))


# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')
