# ---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI----------------------------------------------- #
# ---------------------------------------------SCRIPT OBJECTIVE--------------------------------------------------------- #

# The objective of this script is to derive the eq. of motion of a system by applying kane's algorithm.

# The script will be adjusted case by case by the user, following the instruction prompted on the terminal

# Kane's algorithm stars by defining the position vectors of the center of mass of the bodies, and then
# derive them along time to obtain the velocity and acceleration vectors for the two points.
# We then derive the partial velocity vectors from the total velocity vectors of the two center of mass, and then
# we proceed to calculate the generalized forces acting on our system.
# We then proceed to assemble's kane's eq. of motion to obtain, by symplification, the eq of motion of the system.



import csv
import time
file_path = "tempi_di_esecuzione_doublependulum_4dof_Inteli7.csv"
tempi = []

for i in range(10):


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


   # ------------------------------ STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

   start = time.perf_counter()


   # ---------------------------------------- SYSTEM GENERAL PARAMETERS ------------------------------------------------- #

   n = 2                      # Number of bodies present 
   m_h = 8                   # Number of constraints present 
   m_nh = 0                   # Number of Non-Holonomic constrains present


   # Calculations
   dof = 6*n - m_h            # System's Degree of freedom, tells us the n° of Gen. Coordinates we will have
   n_coor = dof + m_nh        # Number of coordinates we have to define


   # Coordinates definition
   q1 = me.dynamicsymbols('q1')
   q2 = me.dynamicsymbols('q2')
   q3 = me.dynamicsymbols('q3')
   q4 = me.dynamicsymbols('q4')




   # -------------------------------------------- SYMBOLIC VARIABLES --------------------------------------------------- #

   # Initial motion parameters
   t = sy.symbols('t')                # Definition of the time variable
   g = sy.symbols('g')                # Gravitational acceleration [m/s^2]
   m = sy.symbols('m')                # Particles Mass [kg]
   l = sy.symbols('l')                # Rod length [m]

   # Inertia of the cart, along z axis [kg*m^2].
   I_m1 = sy.Matrix([[0,0,0],
                     [0,0,0],
                     [0,0,0]])     

   # Inertia of the pendulum, along z axis [kg*m^2]. 
   I_m2 = sy.Matrix([[0,0,0],
                     [0,0,0],
                     [0,0,0]])     

   # ----------------------------------------- REFERENCE FRAME DEFINITION -------------------------------------------------- #

   # Reference Frame definition
   N = me.ReferenceFrame('N')          # Inertial ref. Frame, fixed with the pavement   
   P1 = me.ReferenceFrame('P1')        # Mass P body ref. Frame, with origin centered on point O
   P2 = me.ReferenceFrame('P2')        # Mass Q body ref. Frame, with origin centered on  point P


   # Reference Frame's origin definition
   O = me.Point('O')                 # Inertial ref. Frame origin
   P = me.Point('P')                 # Mass P body ref. Frame origin
   Q = me.Point('Q')                 # Mass Q body ref. Frame origin


   # Setting the relative position between the frame's origins
   P.set_pos(O, l*P1.y)                                       # Setting point P relative to the pendulum 1 ref. Frame
   O.set_pos(O, 0*P1.x + 0*P1.y + 0*P1.z)                     # Setting point O as the origin of the pendulum 1 ref. frame
   O.set_pos(O, 0*N.x + 0*N.y + 0*N.z)

   Q.set_pos(P, (l)*P2.y)                                     # Setting point Q relative to the pendulum 2 ref. Frame
   P.set_pos(P, 0*P2.x + 0*P2.y + 0*P2.z)                     # Setting point P as the origin of the prendulum 2 ref. frame


   # Setting the relative orientation between the frames
   P1.orient_body_fixed(N, (-(sy.pi/2 - q1),q4,0), 'zxy')                                     #Defining the pendulum 1 ref. frame as rotated of q1 wrt the Inertial. By doing so, i tell the code that the two frames have their axes alligned
   P2.orient_body_fixed(P1, (q2,q3,0), 'zxy')                                   #Defining the pendulum 2 ref. frame as rotated of q2 wrt to pendulum 2 ref. frame


   # Setting the velocity of the inertial ref. frame to 0, to allow the calculations to be developed
   O.set_vel(N, 0)
   N.set_ang_vel(N, 0)



   # ----------------------------------------------VECTORS CALCULATIONS-------------------------------------------------- #

   # Position vector definition
   x_P = P.pos_from(O).express(N)
   x_Q = Q.pos_from(O).express(N)


   # Velocity vector definition
   v_P = P.vel(N)
   v_Q = Q.vel(N)


   # Acceleration vector definition
   a_P = P.acc(N)
   a_Q = Q.acc(N)


   # Angular velocity vector
   omega_P = P1.ang_vel_in(N)
   omega_Q = P2.ang_vel_in(N)


   # Angular acceleration vector definition
   omegadot_P = P1.ang_acc_in(N) 
   omegadot_Q = P2.ang_acc_in(N)



   # ------------------------------------------- GENERALIZED SPEEDS ----------------------------------------------------- #

   # Generalized Speeds definition
   u1 = me.dynamicsymbols('u1')      # <- Linear velocity displacement of the cart center of mass, function of time "t"
   u2 = me.dynamicsymbols('u2')      # <- Angular velocity displacement of the pendulum rod, function of time "t"
   u3 = me.dynamicsymbols('u3')
   u4 = me.dynamicsymbols('u4')


   # # Projected Angular velocities 
   # omega_i = [omega_P.dot(P1.x), omega_P.dot(P1.y), omega_P.dot(P1.z), 
   #            omega_Q.dot(P2.x), omega_Q.dot(P2.y), omega_Q.dot(P2.z)]

   # sy.pprint(omega_i)


   # Dictionary with the substitution to do
   substitutions = {
      
      q1.diff(): u1, 
      q2.diff(): u2 - u1,
      q3.diff(): u3,
      q4.diff(): u4 - u3,

      q1.diff(t,2): u1.diff(),
      q2.diff(t,2): u2.diff() - u1.diff(),
      q3.diff(t,2): u3.diff(),
      q4.diff(t,2): u4.diff() - u3.diff(),

      }     


   # Vectors rewriting
   v_P = v_P.subs(substitutions)
   v_Q = v_Q.subs(substitutions)
   a_P = a_P.subs(substitutions)
   a_Q = a_Q.subs(substitutions)
   omega_P = omega_P.subs(substitutions)
   omega_Q = omega_Q.subs(substitutions)
   omegadot_P = omegadot_P.subs(substitutions)
   omegadot_Q = omegadot_Q.subs(substitutions)



   # ------------------------------------------ PARTIAL VELOCITIES CALCULATIONS--------------------------------------------------- #

   # Partial Velocities for the system
   v_P_part1 = v_P.diff(u1, N)
   v_P_part2 = v_P.diff(u2, N)    
   v_P_part3 = v_P.diff(u3, N)
   v_P_part4 = v_P.diff(u4, N)                                             
   v_Q_part1 = v_Q.diff(u1, N)
   v_Q_part2 = v_Q.diff(u2, N)
   v_Q_part3 = v_Q.diff(u3, N) 
   v_Q_part4 = v_Q.diff(u4, N)                              
   omega_P_part1 = omega_P.diff(u1, N)
   omega_P_part2 = omega_P.diff(u2, N)
   omega_P_part3 = omega_P.diff(u3, N)
   omega_P_part4 = omega_P.diff(u4, N)
   omega_Q_part1 = omega_Q.diff(u1, N)
   omega_Q_part2 = omega_Q.diff(u2, N)
   omega_Q_part3 = omega_Q.diff(u3, N)
   omega_Q_part4 = omega_Q.diff(u4, N)


   # Rearranging the partial velocities into matrices for ease of use
   v_A_part = sy.Matrix([[v_P_part1.dot(N.x), v_P_part1.dot(N.y), v_P_part1.dot(N.z)],
                        [v_P_part2.dot(N.x), v_P_part2.dot(N.y), v_P_part2.dot(N.z)],
                        [v_P_part3.dot(N.x), v_P_part3.dot(N.y), v_P_part3.dot(N.z)],
                        [v_P_part4.dot(N.x), v_P_part4.dot(N.y), v_P_part4.dot(N.z)]])

   v_G_part = sy.Matrix([[v_Q_part1.dot(N.x), v_Q_part1.dot(N.y), v_Q_part1.dot(N.z)],
                        [v_Q_part2.dot(N.x), v_Q_part2.dot(N.y), v_Q_part2.dot(N.z)],
                        [v_Q_part3.dot(N.x), v_Q_part3.dot(N.y), v_Q_part3.dot(N.z)],
                        [v_Q_part4.dot(N.x), v_Q_part4.dot(N.y), v_Q_part4.dot(N.z)]])

   omega_C_part = sy.Matrix([[omega_P_part1.dot(N.x), omega_P_part1.dot(N.y), omega_P_part1.dot(N.z)],
                           [omega_P_part2.dot(N.x), omega_P_part2.dot(N.y), omega_P_part2.dot(N.z)],
                           [omega_P_part3.dot(N.x), omega_P_part3.dot(N.y), omega_P_part3.dot(N.z)],
                           [omega_P_part4.dot(N.x), omega_P_part4.dot(N.y), omega_P_part4.dot(N.z)]])

   omega_P_part = sy.Matrix([[omega_Q_part1.dot(N.x), omega_Q_part1.dot(N.y), omega_Q_part1.dot(N.z)],
                           [omega_Q_part2.dot(N.x), omega_Q_part2.dot(N.y), omega_Q_part2.dot(N.z)],
                           [omega_Q_part3.dot(N.x), omega_Q_part3.dot(N.y), omega_Q_part3.dot(N.z)],
                           [omega_Q_part4.dot(N.x), omega_Q_part4.dot(N.y), omega_Q_part4.dot(N.z)]])



   # ----------------------------------------- FORCES & TORQUES ACTING ON OUR BODIES---------------------------------------------- #

   # For now, specifying the point in which the force vector is applied doesnt change anything. I hope in the future to make the code able to derive the equivalent
   # force system applied to the c.o.m of the bodies, specifying the forces on the position in which are applied.

   # Forces acting on point P
   f_1 = (m*g)*N.x

   F_1 = (P, f_1)


   # Forces acting on point Q
   f_2 = (m*g)*N.x
   F_2 = (Q, f_2)


   # Writing the vectors as matrices for easy of use
   F_1_m = sy.Matrix([F_1[1].dot(N.x), F_1[1].dot(N.y), F_1[1].dot(N.z)])
   F_2_m = sy.Matrix([F_2[1].dot(N.x), F_2[1].dot(N.y), F_2[1].dot(N.z)])



   # Torques acting on the pendulum P
   t_1 = 0*N.z
   T_1 = (P, t_1)


   # Torques acting on the pendulum Q
   t_2 = 0*N.z
   T_2 = (Q, t_2)


   # Writing the vectors are matrices for easy of use
   T_1_m = sy.Matrix([T_1[1].dot(N.x), T_1[1].dot(N.y), T_1[1].dot(N.z)])
   T_2_m = sy.Matrix([T_2[1].dot(N.x), T_2[1].dot(N.y), T_2[1].dot(N.z)])


   # Calculating the resultant forces & torques acting on the two bodies
   Ftot_1 = F_1_m                               #Total force vector acting on the pendulum P
   Torqtot_1 = T_1_m                            #Total torque vector acting on the pendulum P

   Ftot_2 = F_2_m                               #Total force vector acting on the pendulum Q
   Torqtot_2 = T_2_m                            #Total torque vector acting on the pendulum Q



   # ---------------------------------------- GENERALIZED INERTIA & ACTIVE FORCES----------------------------------------- #

   # Rearranging quantities into matrices for ease of use

   # Angular velocity matrix for the entire system
   omega = sy.Matrix([[omega_P.dot(N.x), omega_P.dot(N.y), omega_P.dot(N.z)],
                     [omega_Q.dot(N.x), omega_Q.dot(N.y), omega_Q.dot(N.z)]])    

   # Acceleration matrix for the entire system
   a = sy.Matrix([[a_P.dot(N.x), a_P.dot(N.y), a_P.dot(N.z)],
                  [a_Q.dot(N.x), a_Q.dot(N.y), a_Q.dot(N.z)]])    

   # Angular acceleration matrix for the entire system
   omegadot = sy.Matrix([[omegadot_P.dot(N.x), omegadot_P.dot(N.y), omegadot_P.dot(N.z)],
                        [omegadot_Q.dot(N.x), omegadot_Q.dot(N.y), omegadot_Q.dot(N.z)]])

   # Mass vector for the entire system
   Md = sy.Matrix([m, m])

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



   # Printing on screen the eq. of motion derived
   # sy.pprint((K[0]))
   # print()
   # print()
   # sy.pprint((K[1]))
   # print()
   # print()
   # sy.pprint((K[2]))
   # print()
   # print()
   # sy.pprint((K[3]))
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