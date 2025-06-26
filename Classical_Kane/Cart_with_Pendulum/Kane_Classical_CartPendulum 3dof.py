#--------------------------------- SCRIPT TEST PYTHON - FRANCESCO MARRADI -----------------------------------------------#
#--------------------------------------------- SCRIPT OBJECTIVE ---------------------------------------------------------#

# The objective of this script is to derive the eq. of motion of a system by applying kane's algorithm.

# Kane's algorithm stars by defining the position vectors of the center of mass of the bodies, and then
# derive them along time to obtain the velocity and acceleration vectors for the two points.
# We then derive the partial velocity vectors from the total velocity vectors of the two center of mass, and then
# we proceed to calculate the generalized forces acting on our system.
# We then proceed to assemble's kane's eq. of motion to obtain, by symplification, the eq of motion of the system.


import csv
import time
file_path = "tempi_di_esecuzione_cartpendulum_3dof_Inteli7.csv"
tempi = []

for i in range(10):


   #------------------------------------ IMPORTING THE NECESSARY PACKAGES -------------------------------------------------#

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

   n = 2                      # Number of bodies present 
   m_h = 9                   # Number of constraints present 
   m_nh = 0                   # Number of Non-Holonomic constrains present


   # Calculations
   dof = 6*n - m_h            # System's Degree of freedom, tells us the n° of Gen. Coordinates we will have
   n_coor = dof + m_nh        # Number of coordinates we have to define


   # Coordinates definition
   q1 = me.dynamicsymbols('q1')
   q2 = me.dynamicsymbols('q2')
   q3 = me.dynamicsymbols('q3')


   # ---------------------------------------..--- SYMBOLIC VARIABLES ---------------------------------------------------- #

   # Initial motion parameters
   t = sy.symbols('t')                # Definition of the time variable
   g = sy.symbols('g')                # Gravitational acceleration [m/s^2]
   m1 = sy.symbols('m1')              # Mass of body 1 [kg]
   m2 = sy.symbols('m2')              # Mass of body 2 [kg]
   b = sy.symbols('b')                # Damper coefficient [N*s/m]
   k = sy.symbols('k')                # spring constant [N/m]
   l = sy.symbols('l')                # Rod length [m]
   h = sy.symbols('h')                # Box height [m]
   w = sy.symbols('h')                # Box width [m]
   a = sy.symbols('a')                # Cart center of mass's height wrt x axis [m] 


   # Inertia of the cart, along z axis [kg*m^2].
   I_m1 = sy.Matrix([[(1/12)*m1*(w**2 + 0.7*w**2),0,0],
                     [0,(1/12)*m1*(h**2 + 0.7*w**2),0],
                     [0,0,(1/12)*m1*(h**2 + w**2)]])     

   # Inertia of the pendulum, along z axis [kg*m^2]. 
   I_m2 = sy.Matrix([[(1/3)*m2*(l**2),0,0],
                     [0,(1/3)*m2*(l**2),0],
                     [0,0,(1/3)*m2*(l**2)]])           



   # -----------------------------------------REFERENCE FRAME DEFINITION-------------------------------------------------- #

   # Reference Frame definition
   N = me.ReferenceFrame('N')        # Inertial ref. Frame, fixed with the pavement   
   C = me.ReferenceFrame('C')        # Cart body ref. Frame, with origin centered on point A
   P = me.ReferenceFrame('P')        # Pendulum body ref. Frame, with origin centered on  point G


   # Reference Frame's origin definition
   O = me.Point('O')                 # Inertial ref. Frame origin
   A = me.Point('A')                 # Cart ref. Frame origin
   G = me.Point('G')                 # Pendulum body ref. Frame origin


   # Setting the relative position between the frame's origins
   A.set_pos(O, q1*N.x + a*N.y)                               # Setting point A relative to the inertial ref. Frame
   A.set_pos(A, 0*C.x + 0*C.y + 0*C.z)                        # Setting point A as the origin of the cart ref. frame

   G.set_pos(A, (l/2)*P.x)                                    # Setting point A relative to the cart ref. Frame
   G.set_pos(G, 0*P.x + 0*P.y + 0*P.z)                        # Setting point G as the origin of the prendulum ref. frame


   # Setting the relative orientation between the frames
   C.orient_axis(N, 0, N.z)                                     #Defining the cart ref. frame as rotated of 0° wrt the Inertial. By doing so, i tell the code that the two frames have their axes alligned
   # P.orient_axis(C, -(sy.pi/2 - q2), C.z)                                    #Defining the pendulum ref. frame as rotated of q2 wrt to the cart ref. frame
   P.orient_body_fixed(C, (-(sy.pi/2 - q2),q3,0),'zyx')

   # Setting the velocity of the inertial ref. frame to 0, to allow the calculations to be developed
   O.set_vel(N, 0)
   N.set_ang_vel(N, 0)



   # ----------------------------------------------VECTORS CALCULATIONS-------------------------------------------------- #

   # Position vector definition
   x_A = A.pos_from(O)
   x_G = G.pos_from(O)


   # Velocity vector definition
   v_A = A.vel(N)
   v_G = G.vel(N)


   # Acceleration vector definition
   a_A = A.acc(N)
   a_G = G.acc(N)


   # Angular velocity vector
   omega_C = C.ang_vel_in(N)
   omega_P = P.ang_vel_in(N)


   # Angular acceleration vector definition
   omegadot_C = C.ang_acc_in(N) 
   omegadot_P = P.ang_acc_in(N)



   # ------------------------------------------- GENERALIZED SPEEDS ----------------------------------------------------- #

   # Generalized Speeds definition
   u1 = me.dynamicsymbols('u1')      # <- Linear velocity displacement of the cart center of mass, function of time "t"
   u2 = me.dynamicsymbols('u2')      # <- Angular velocity displacement of the pendulum rod, function of time "t"
   u3 = me.dynamicsymbols('u2')      # <- Angular velocity displacement of the pendulum rod, function of time "t"


   # # Projected Angular velocities 
   # omega_i = [omega_C.dot(C.x), omega_C.dot(C.y), omega_C.dot(C.z), 
   #            omega_P.dot(P.x), omega_P.dot(P.y), omega_P.dot(P.z)]

   # sy.pprint(omega_i)


   # Dictionary with the substitution to do
   substitutions = {
      
      q1.diff(): u1, 
      q2.diff(): u2,
      q3.diff(): u3,
      q1.diff(t,2): u1.diff(),
      q2.diff(t,2): u2.diff(),
      q3.diff(t,2): u3.diff(),

      }     


   # Vectors rewriting
   v_A = v_A.subs(substitutions)
   v_G = v_G.subs(substitutions)
   a_A = a_A.subs(substitutions)
   a_G = a_G.subs(substitutions)
   omega_C = omega_C.subs(substitutions)
   omega_P = omega_P.subs(substitutions)
   omegadot_C = omegadot_C.subs(substitutions)
   omegadot_P = omegadot_P.subs(substitutions)



   # ------------------------------------------ PARTIAL VELOCITIES CALCULATIONS--------------------------------------------------- #

   # Partial Velocities for the system
   v_A_part1 = v_A.diff(u1, N)
   v_A_part2 = v_A.diff(u2, N)
   v_A_part3 = v_A.diff(u3, N)                                                  
   v_G_part1 = v_G.diff(u1, N)
   v_G_part2 = v_G.diff(u2, N) 
   v_G_part3 = v_G.diff(u3, N)                             
   omega_C_part1 = omega_C.diff(u1, N)
   omega_C_part2 = omega_C.diff(u2, N)
   omega_C_part3 = omega_C.diff(u3, N)
   omega_P_part1 = omega_P.diff(u1, N)
   omega_P_part2 = omega_P.diff(u2, N)
   omega_P_part3 = omega_P.diff(u3, N)


   # Rearranging the partial velocities into matrices for ease of use
   v_A_part = sy.Matrix([[v_A_part1.dot(N.x), v_A_part1.dot(N.y), v_A_part1.dot(N.z)],
                        [v_A_part2.dot(N.x), v_A_part2.dot(N.y), v_A_part2.dot(N.z)],
                        [v_A_part3.dot(N.x), v_A_part3.dot(N.y), v_A_part3.dot(N.z)]])

   v_G_part = sy.Matrix([[v_G_part1.dot(N.x), v_G_part1.dot(N.y), v_G_part1.dot(N.z)],
                        [v_G_part2.dot(N.x), v_G_part2.dot(N.y), v_G_part2.dot(N.z)],
                        [v_G_part3.dot(N.x), v_G_part3.dot(N.y), v_G_part3.dot(N.z)]])


   omega_C_part = sy.Matrix([[omega_C_part1.dot(N.x), omega_C_part1.dot(N.y), omega_C_part1.dot(N.z)],
                           [omega_C_part2.dot(N.x), omega_C_part2.dot(N.y), omega_C_part2.dot(N.z)],
                           [omega_C_part3.dot(N.x), omega_C_part3.dot(N.y), omega_C_part3.dot(N.z)]])

   omega_P_part = sy.Matrix([[omega_P_part1.dot(N.x), omega_P_part1.dot(N.y), omega_P_part1.dot(N.z)],
                           [omega_P_part2.dot(N.x), omega_P_part2.dot(N.y), omega_P_part2.dot(N.z)],
                           [omega_P_part3.dot(N.x), omega_P_part3.dot(N.y), omega_P_part3.dot(N.z)]])



   # ----------------------------------------- FORCES & TORQUES ACTING ON OUR BODIES---------------------------------------------- #

   # For now, specifying the point in which the force vector is applied doesnt change anything. I hope in the future to make the code able to derive the equivalent
   # force system applied to the c.o.m of the bodies, specifying the forces on the position in which are applied.

   # Forces acting on point A, com of the cart
   f_1 = (-k*q1)*C.x
   f_2 = (-b*u1)*C.x
   f_3 = (-m1*g)*C.y

   F_1 = (A, f_1)
   F_2 = (A, f_2)
   F_3 = (A, f_3)


   # Forces acting on point G, com of the pendulum
   f_4 = (-m2*g)*N.y
   F_4 = (P, f_4)


   # Writing the vectors as matrices for easy of use
   F_1_m = sy.Matrix([F_1[1].dot(N.x), F_1[1].dot(N.y), F_1[1].dot(N.z)])
   F_2_m = sy.Matrix([F_2[1].dot(N.x), F_2[1].dot(N.y), F_2[1].dot(N.z)])
   F_3_m = sy.Matrix([F_3[1].dot(N.x), F_3[1].dot(N.y), F_3[1].dot(N.z)])
   F_4_m = sy.Matrix([F_4[1].dot(N.x), F_4[1].dot(N.y), F_4[1].dot(N.z)])


   # Torques acting on the cart C
   t_1 = 0*N.z
   T_1 = (C, t_1)


   # Torques acting on the pendulum P
   t_2 = 0*N.z
   T_2 = (P, t_2)


   # Writing the vectors are matrices for easy of use
   T_1_m = sy.Matrix([T_1[1].dot(N.x), T_1[1].dot(N.y), T_1[1].dot(N.z)])
   T_2_m = sy.Matrix([T_2[1].dot(N.x), T_2[1].dot(N.y), T_2[1].dot(N.z)])


   # Calculating the resultant forces & torques acting on the two bodies
   Ftot_1 = F_1_m + F_2_m + F_3_m               #Total force vector acting on the cart
   Torqtot_1 = T_1_m                            #Total torque vector acting on the cart

   Ftot_2 = F_4_m                               #Total force vector acting on the pendulum
   Torqtot_2 = T_1_m                            #Total torque vector acting on the pendulum



   # ---------------------------------------- GENERALIZED INERTIA & ACTIVE FORCES----------------------------------------- #

   # Rearranging quantities into matrices for ease of use

   # Angular velocity matrix for the entire system
   omega = sy.Matrix([[omega_C.dot(N.x), omega_C.dot(N.y), omega_C.dot(N.z)],
                     [omega_P.dot(N.x), omega_P.dot(N.y), omega_P.dot(N.z)]])    

   # Acceleration matrix for the entire system
   a = sy.Matrix([[a_A.dot(N.x), a_A.dot(N.y), a_A.dot(N.z)],
                  [a_G.dot(N.x), a_G.dot(N.y), a_G.dot(N.z)]])    

   # Angular acceleration matrix for the entire system
   omegadot = sy.Matrix([[omegadot_C.dot(N.x), omegadot_C.dot(N.y), omegadot_C.dot(N.z)],
                        [omegadot_P.dot(N.x), omegadot_P.dot(N.y), omegadot_P.dot(N.z)]])

   # Mass vector for the entire system
   Md = sy.Matrix([m1, m2])

   # Force Matrix
   Ftot = sy.Matrix([[Ftot_1.T],
                     [Ftot_2.T]])    

   # Torque Matrix
   Torqtot =  sy.Matrix([[Torqtot_1.T],
                        [Torqtot_2.T]]) 



   # Initializing the empty vectors
   F = sy.Matrix.zeros(dof,1)         # Generalized inertia forces vertical vector
   f = sy.Matrix.zeros(dof,1)         # Generalized forces vertical vector
   K = sy.Matrix.zeros(dof,1)         # Kane's eq. vector vertical vector


   # Calculating the Generalized forces with a for cycle
   for i in range(dof):

      # Generalized Inertia forces
      F[i,0] = - (Md[0] * a[0,0:3].dot(v_A_part[i,0:3]) + ((omegadot[0,0:3]*I_m1) + omega[0,0:3].cross(I_m1*(omega[0,0:3].T))).dot(omega_C_part[i,0:3])) - (Md[1] * a[1,0:3].dot(v_G_part[i,0:3]) + ((omegadot[1,0:3]*I_m2) + omega[1,0:3].cross(I_m2*(omega[1,0:3].T))).dot(omega_P_part[i,0:3]))

      # Generalized Active forces
      f[i,0] = (Ftot[0,0:3].dot(v_A_part[i,0:3]) + (Torqtot[0,0:3]).dot(omega_C_part[i,0:3])) + (Ftot[1,0:3].dot(v_G_part[i,0:3]) + (Torqtot[1,0:3]).dot(omega_P_part[i,0:3]))



   # ------------------------------------------- ASSEMBLY KANE'S EQ. OF MOTION------------------------------------------- #

   # Assembly Kane's Eq. of Motion
   for i in range(dof):
      
      K[i,0] = F[i,0] + f[i,0]



   # # Printing on screen the eq. of motion derived
   # sy.pprint(sy.simplify(K[0]))
   # print()
   # print()
   # sy.pprint(sy.simplify(K[1]))
   # print()
   # print()
   # sy.pprint(sy.simplify(K[2]))
   # print()

   # ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

   end = time.perf_counter()

   # Showing the computation time
   print(f"The calculations required time was: {end - start:.4f} seconds")
   tempi.append(end - start)


# Salva su CSV
with open(file_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Durata (s)"])
    for i, tempo in enumerate(tempi):
        writer.writerow([i + 1, round(tempo, 4)])

print(f"\nTempi salvati su '{file_path}'")


print()
print('Codice terminato')
