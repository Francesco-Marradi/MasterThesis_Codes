import csv
import time
file_path = "tempi_di_esecuzione_3Dfakependulum_6dof_Inteli7.csv"
tempi = []

for i in range(10):

    # ------------------------------------ IMPORTING THE NECESSARY PACKAGES ------------------------------------------------- #

    import sympy as sy
    import sympy.physics.mechanics as me
    import matplotlib.pyplot as plt
    import os
    import matplotlib.pyplot as plt
    import time




    # Setting the enviroment variable
    os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

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


    # ---------------------------------------------- SYMBOLIC VARIABLES ---------------------------------------------------- #    

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

    # Joints Ref. Frame origin definitions
    G0 = me.Point('G0')                 # Joint G0
    G1 = me.Point('G1')                 # Joint G1



    # Setting the relative position between the frame's origins
    B0.set_pos(O, q6*n.x + q5*n.y + 0*n.z)                         # Setting point B0 relative to the inertial ref. Frame
    B0.set_pos(B0, 0*b0.x + 0*b0.y + 0*b0.z)                     # Setting point B0 as the origin of the root ref. frame

    B1.set_pos(B0, (l/2)*b1.x + 0*b1.y + 0*b1.z)               # Setting the point B1 relative to the joint position
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
    b1.orient_body_fixed(b0, (q1, 0, q4), 'zxy')

    # Defining the pendulum ref. frame as rotated of 0° wrt to the joint ref. frame                                
    b2.orient_body_fixed(b1, (q3, q2, 0), 'zxy')


    # Setting the velocity of the inertial ref. frame to 0, to allow the calculations to be developed
    O.set_vel(n, 0)
    n.set_ang_vel(n, 0)



    # Mass List
    M_list = [0, m, m]


    # Inertia Dyadic for each body, centered in their respective com and expressed along their principal inertia axis
    I_0 = (0)*m*(l**2)*(me.outer(b0.x, b0.x) + me.outer(b0.y, b0.y) + me.outer(b0.z, b0.z))
    I_1 = (1/12)*m*(l**2)*(0*me.outer(b1.x, b1.x) + me.outer(b1.y, b1.y) + me.outer(b1.z, b1.z))
    I_2 = (1/12)*m*(l**2)*(me.outer(b2.x, b2.x) + 0*me.outer(b2.y, b2.y) + me.outer(b2.z, b2.z))


    # Inertia List
    I_list = [I_0, I_1, I_2]



    # # -------------------------------------------------- BLOCKING DOF ---------------------------------------------------- #

    # # Here we want to "block" the dof of the system. Right now the problem should be defined in the most general way possible
    # # In case there are dof blocked, like the motion is planar... we should "tell" the code right now. In this way, we can 
    # # safe some computational time, istead of letting the code run with the most general version of the problem and turn off 
    # # unnecessary dof at the end 

    # # We will use a dictionary that will set to zero each gen. coordinates/speeds, blocking that dof for the system. 
    # gencoord_active = sy.Matrix([0, 0, 0, 0, 0, q6, q7, 0, 0, 0, 0, 0])
    # genspeeds_active = sy.Matrix([0, 0, 0, 0, 0, u6, u7, 0, 0, 0, 0, 0])

    # # Dictionary to turn off the dof of the system
    # gencoord_dict = dict(zip(BdStVec_list_gen ,gencoord_active))
    # genspeeds_dict = dict(zip(StVec_list_gen ,genspeeds_active))



    # ------------------------------------------- CLASSICAL APPROACH ---------------------------------------------------- #


    v_Ao_1 = B1.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u1, n)
    v_Ao_2 = B1.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u2, n)
    v_Ao_3 = B1.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u3, n)
    v_Ao_4 = B1.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u4, n)
    v_Ao_5 = B1.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u5, n)
    v_Ao_6 = B1.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u6, n)
    v_Bo_1 = B2.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u1, n)
    v_Bo_2 = B2.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u2, n)
    v_Bo_3 = B2.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u3, n)
    v_Bo_4 = B2.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u4, n)
    v_Bo_5 = B2.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u5, n)
    v_Bo_6 = B2.vel(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u6, n)
    w_A_1 = b1.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u1, n)
    w_A_2 = b1.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u2, n)
    w_A_3 = b1.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u3, n)
    w_A_4 = b1.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u4, n)
    w_A_5 = b1.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u5, n)
    w_A_6 = b1.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u6, n)
    w_B_1 = b2.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u1, n)
    w_B_2 = b2.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u2, n)
    w_B_3 = b2.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u3, n)
    w_B_4 = b2.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u4, n)
    w_B_5 = b2.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u5, n)
    w_B_6 = b2.ang_vel_in(n).xreplace(gencoordDer_to_speeds_dict).xreplace(gencoordDerDer_to_speedsDer_dict).diff(u6, n)

    Rs_Ao = -m*B1.acc(n)
    Rs_Bo = -m*B2.acc(n)
    R_Ao = (m*g*n.x)
    R_Bo = (m*g*n.x)


    Ts_A = -(b1.ang_acc_in(n).dot(I_1) + me.cross(b1.ang_vel_in(n), I_1).dot(b1.ang_vel_in(n)))
    Ts_B = -(b2.ang_acc_in(n).dot(I_2) + me.cross(b2.ang_vel_in(n), I_2).dot(b2.ang_vel_in(n)))
    T_A = (-k*q1*n.z + k*q2*b1.x)
    T_B = (-k*q2*b1.x)


    F1s_A = v_Ao_1.dot(Rs_Ao) + w_A_1.dot(Ts_A)
    F1s_B = v_Bo_1.dot(Rs_Bo) + w_B_1.dot(Ts_B)
    F2s_A = v_Ao_2.dot(Rs_Ao) + w_A_2.dot(Ts_A)
    F2s_B = v_Bo_2.dot(Rs_Bo) + w_B_2.dot(Ts_B)
    F3s_A = v_Ao_3.dot(Rs_Ao) + w_A_3.dot(Ts_A)
    F3s_B = v_Bo_3.dot(Rs_Bo) + w_B_3.dot(Ts_B)
    F4s_A = v_Ao_4.dot(Rs_Ao) + w_A_4.dot(Ts_A)
    F4s_B = v_Bo_4.dot(Rs_Bo) + w_B_4.dot(Ts_B)
    F5s_A = v_Ao_5.dot(Rs_Ao) + w_A_5.dot(Ts_A)
    F5s_B = v_Bo_5.dot(Rs_Bo) + w_B_5.dot(Ts_B)
    F6s_A = v_Ao_6.dot(Rs_Ao) + w_A_6.dot(Ts_A)
    F6s_B = v_Bo_6.dot(Rs_Bo) + w_B_6.dot(Ts_B)

    F1_A = v_Ao_1.dot(R_Ao) + w_A_1.dot(T_A)
    F1_B = v_Bo_1.dot(R_Bo) + w_B_1.dot(T_B)
    F2_A = v_Ao_2.dot(R_Ao) + w_A_2.dot(T_A)
    F2_B = v_Bo_2.dot(R_Bo) + w_B_2.dot(T_B)
    F3_A = v_Ao_3.dot(R_Ao) + w_A_3.dot(T_A)
    F3_B = v_Bo_3.dot(R_Bo) + w_B_3.dot(T_B)
    F4_A = v_Ao_4.dot(R_Ao) + w_A_4.dot(T_A)
    F4_B = v_Bo_4.dot(R_Bo) + w_B_4.dot(T_B)
    F5_A = v_Ao_5.dot(R_Ao) + w_A_5.dot(T_A)
    F5_B = v_Bo_5.dot(R_Bo) + w_B_5.dot(T_B)
    F6_A = v_Ao_6.dot(R_Ao) + w_A_6.dot(T_A)
    F6_B = v_Bo_6.dot(R_Bo) + w_B_6.dot(T_B)


    F1s = F1s_A + F1s_B
    F2s = F2s_A + F2s_B
    F3s = F3s_A + F3s_B
    F4s = F4s_A + F4s_B
    F5s = F5s_A + F5s_B
    F6s = F6s_A + F6s_B
    F1 = F1_A + F1_B
    F2 = F2_A + F2_B
    F3 = F3_A + F3_B
    F4 = F4_A + F4_B
    F5 = F5_A + F5_B
    F6 = F6_A + F6_B


    # print()
    # sy.pprint((F1s + F1))
    # print()
    # print()
    # sy.pprint((F2s + F2))
    # print()
    # print()
    # sy.pprint((F3s + F3))
    # print()
    # print()
    # sy.pprint((F4s + F4))

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
