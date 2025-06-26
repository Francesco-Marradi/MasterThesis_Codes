# ---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI----------------------------------------------- #
# ---------------------------------------------SCRIPT OBJECTIVE--------------------------------------------------------- #

# The objective of this script is to derive the eq. of motion of the "Double Pendulum" system by 
# applying kane's algorithm.

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
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import os
import matplotlib.pyplot as plt
import time

#Cleaning the terminal and closing all open graphics
os.system('cls')
plt.close('all')


# ---------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

start = time.perf_counter()


# ---------------------------------------- SYSTEM GENERAL PARAMETERS ------------------------------------------------- #

n = 2                      # Number of bodies present 
m_c = 10                   # Number of constraints present 
m_nh = 0                   # Number of Non-Holonomic constrains present

dof = 6*n - m_c            # System's Degree of freedom
n_gen_coor = dof + m_nh    # Number of generalized coordinates necessary



# ------------------------------------------ SYMBOLIC VARIABLES ------------------------------------------------------ #

'''
Every term of the equations will be treated as a symbolic variable, only when we need to numerically solve the eq.
of motion, we will substitute the numeric value
'''
# Initial motion parameters
g = sy.symbols('g')         # Gravitational acceleration [m/s^2]
mp = sy.symbols('mp')       # Mass of point P [kg]
mq = sy.symbols('mq')       # Mass of point Q [kg]
l = sy.symbols('l')         # Rods length [m]
t = sy.symbols('t')         # Time [s]
t_max = 10                  # Maximum simulation calculation time [s]


# Coordinates definition
q1 = me.dynamicsymbols('q1')        # Angular displacement of the firts pendulum rod, function of time "t"
q2 = me.dynamicsymbols('q2')        # Angular displacement of the second pendulum rod, function of time "t"



# -----------------------------------------REFERENCE FRAME DEFINITION-------------------------------------------------- #

'''
Let's start by defining an inertial reference frame N, and a n° of additional ref. frame equal to the n° of bodies our
system has.
'''
# Reference Frame definition
N = me.ReferenceFrame('N')       # Inertial ref. Frame, fixed with the pavement   
P1 = me.ReferenceFrame('P1')      # First pendulum body ref. Frame, with origin centered on point P
P2 = me.ReferenceFrame('P2')      # Second Pendulum body ref. Frame, with origin centered on  point Q


'''
Now we define the origin of our reference frames.
We need to refer the ref. frame one to the other, more syecifically, defining the position vectors from point O->P 
and P->Q, and also defining the relative orientation between the ref. Frame P and Q wrt the inertial one N.
If not syecified, the position of every point we define are centered on the coordinates (0,0,0), the origin of our
ref. system.
'''
# Reference Frame's origin definition
O = me.Point('O')               # Inertial ref. Frame origin
P = me.Point('P')               # First pendulum ref. Frame origin, centered on the particle (P)
Q = me.Point('Q')               # Second Pendulum body ref. Frame origin, centered on the particle (Q)


# Setting the relative position between the frame's origins
P.set_pos(O, l*P1.x)                             #Setting point P relative to the inertial ref. Frame
Q.set_pos(P, l*P2.x)                             #Setting point Q relative to the P wrt P1 ref. frame


# Setting the relative orientation between the frames
P1.orient_axis(N, q1, N.z)                                                    #Defining the cart ref. frame as rotated of 0 wrt the Inertial. By doing so, i tell the code that the two frames have their axes alligned
P2.orient_axis(P1, q2, P1.z)                                                  #Defining the pendulum ref. frame as rotated of -(pi/2 - q2) wrt to the cart ref. frame


# Setting the velocity of the inertial ref. frame to 0
O.set_vel(N, 0)
N.set_ang_vel(N, 0)



# ----------------------------------------------VECTORS CALCULATIONS-------------------------------------------------- #

'''
Moving on, we define the vectors between the points (that are the systems origin's) we definided before. These points
coincides with the c.o.m. of the bodies, therefore, we are calculating the position vector of the c.om. of the bodies 
wrt to inertial ref. frame.
Then, we will calculate the total velocity vectors for the c.o.m. of the bodies and their angular velocity vectors.
Since the body ref. frame are fixed on their bodies, the angular velocity of them is equal to the total angular velocity
of their ref. frames.
'''
# Position vector definition
x_P = P.pos_from(O)
x_Q = Q.pos_from(O)


# Velocity vector definition
v_P = P.vel(N)
v_Q = Q.vel(N)


# Angular velocity vector
omega_P = P1.ang_vel_in(N)
omega_Q = P2.ang_vel_in(N)


# Total acceleration vector definition
a_P = P.acc(N)
a_Q = Q.acc(N)


# Total angular acceleration vector definition
omegadot_P = P1.ang_acc_in(N) 
omegadot_Q = P2.ang_acc_in(N)



# ------------------------------------------- GENERALIZED SPEEDS ----------------------------------------------------- #

'''
To correctly choose the generalized speeds for our system, we show the total velocity vectors projected along the inertial
ref. frame and total angular velocity vectors projected along the bodies ref. frames. The angular velocies are projected
along a differente ref. frame because from Banjeere book we know that this is the smartest way to choose our gen. speeds. 
Projecting them allow to easily indetify them into the equations. As we will see, most of them are equal to 0, that's caused 
by the constraints presents in our system.
'''

# Generalized Speeds definition
u1 = me.dynamicsymbols('u1')      # <- Linear velocity displacement of the cart center of mass, function of time "t"
u2 = me.dynamicsymbols('u2')      # <- Angular velocity displacement of the pendulum rod, function of time "t"


# Linear velocities and projected angular velocities
ui = [v_P.dot(N.x), v_P.dot(N.y), v_P.dot(N.z), v_Q.dot(N.x), v_Q.dot(N.y), v_Q.dot(N.z)]
omega_i = [omega_P.dot(P1.x), omega_P.dot(P1.y), omega_P.dot(P1.z), omega_Q.dot(P2.x), omega_Q.dot(P2.y), omega_Q.dot(P2.z)]



'''
Choose the desired generalized speeds, to do so, inspect the following equations
The number of gen. speeds is equal to the number of gen. coordinates necessary to describe the system''s motion
Define in the code as much gen. speeds as necessary, then write the dictionary for the substitution in the equations
Remember, to call a first time derivative of a function use ".diff()", while for a second time derivative use ".diff(t,2)
'''


# User inputs
sy.pprint(ui)
sy.pprint(omega_i)


# Dictionary with the substitution to do
substitutions = {
     
    q1.diff(): u1, 
    q2.diff(): u2,
    q1.diff(t,2): u1.diff(),
    q2.diff(t,2): u2.diff(),

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


# ----------------------------------------- PARTIAL VELOCITIES CALCULATIONS--------------------------------------------------- #

'''
Next step is to define the partial velocities for our system. To do so, we need to get the terms that multiply the
generalized velocities inside total velocities vectors. we know that this terms are linearly multypling the generalized
speeds, so we can simply partially derive the total velocity vectors for the gen. speeds to get these terms.
'''

# Partial Velocities for the system
v_P_part1 = v_P.diff(u1, N)
v_P_part2 = v_P.diff(u2, N)                                                
v_Q_part1 = v_Q.diff(u1, N)
v_Q_part2 = v_Q.diff(u2, N)                              
omega_P_part1 = omega_P.diff(u1, N)
omega_P_part2 = omega_P.diff(u2, N)
omega_Q_part1 = omega_Q.diff(u1, N)
omega_Q_part2 = omega_Q.diff(u2, N)


# Rearranging the partial velocities into matrices for ease of use
v_P_part = sy.Matrix([[v_P_part1.dot(N.x), v_P_part1.dot(N.y), v_P_part1.dot(N.z)],[v_P_part2.dot(N.x), v_P_part2.dot(N.y), v_P_part2.dot(N.z)]])
v_Q_part = sy.Matrix([[v_Q_part1.dot(N.x), v_Q_part1.dot(N.y), v_Q_part1.dot(N.z)],[v_Q_part2.dot(N.x), v_Q_part2.dot(N.y), v_Q_part2.dot(N.z)]])
omega_P_part = sy.Matrix([[omega_P_part1.dot(N.x), omega_P_part1.dot(N.y), omega_P_part1.dot(N.z)],[omega_P_part2.dot(N.x), omega_P_part2.dot(N.y), omega_P_part2.dot(N.z)]])
omega_Q_part = sy.Matrix([[omega_Q_part1.dot(N.x), omega_Q_part1.dot(N.y), omega_Q_part1.dot(N.z)],[omega_Q_part2.dot(N.x), omega_Q_part2.dot(N.y), omega_Q_part2.dot(N.z)]])



# ----------------------------------------- FORCES & TORQUES ACTING ON OUR BODIES---------------------------------------------- #

'''
We then define the External forces acting on our system, referred to the center of masses of the two bodies and 
projected along the inertial reference system.
Pay attention to the fact that every vector from this line on will be represented as a Matrix [1xn], that beacause
we can use the sympy toolbox to make us able to derive and manage the vectors like we are able to do with pen and papers
'''
# Forces & Torques acting on particle P 
Fp_1 = sy.Matrix([mp*g, 0, 0])


# Forces & Torques acting on body 2 (Pendulum)
Fp_2 = sy.Matrix([mq*g, 0, 0])
Tp_2 = sy.Matrix([0, 0, 0])


# Calculating the resultant forces & torques acting on the two bodies
Ftot_1 = Fp_1                              #Total force vector acting on the cart
Torqtot_1 = sy.Matrix([0,0,0])             #Total torque vector acting on the cart

Ftot_2 = Fp_2                              #Total force vector acting on the pendulum
Torqtot_2 = sy.Matrix([0,0,0])             #Total torque vector acting on the pendulum



# -------------------------------------------- GENERALIZED INERTIA & ACTIVE FORCES--------------------------------------------- #

# Rearranging quantities into matrices for ease of use
# Angular velocity matrix for the entire system
omega = sy.Matrix([[omega_P.dot(N.x), omega_P.dot(N.y), omega_P.dot(N.z)],[omega_Q.dot(N.x), omega_Q.dot(N.y), omega_Q.dot(N.z)]])    

# Acceleration matrix for the entire system
a = sy.Matrix([[a_P.dot(N.x), a_P.dot(N.y), a_P.dot(N.z)],[a_Q.dot(N.x), a_Q.dot(N.y), a_Q.dot(N.z)]])    

# Angular acceleration matrix for the entire system
omegadot = sy.Matrix([[omegadot_P.dot(N.x), omegadot_P.dot(N.y), omegadot_P.dot(N.z)],[omegadot_Q.dot(N.x), omegadot_Q.dot(N.y), omegadot_Q.dot(N.z)]])


# Mass matrix for the entire system
M = sy.Matrix([mp, mq])


# Force Matrix
Ftot = sy.Matrix([[Ftot_1.T],[Ftot_2.T]])    


# Torque Matrix
Torqtot =  sy.Matrix([[Torqtot_1.T],[Torqtot_2.T]]) 


# Initializing the Generalized Forces vectors
F = [0, 0]
f = [0, 0]
K = [0, 0]



# Calculating the Generalized forces with a for cycle
for i in range(2):

   # Generalized Inertia forces
   F[i] = - M[0] * a[0,0:3].dot(v_P_part[i,0:3]) - M[1] * a[1,0:3].dot(v_Q_part[i,0:3]) 

   # Generalized Active forces
   f[i] = M[0] * Ftot[0,0:3].dot(v_P_part[i,0:3]) + M[1] * Ftot[1,0:3].dot(v_Q_part[i,0:3])



# -------------------------------------------- ASSEMBLY KANE'S EQ. OF MOTION--------------------------------------------- #

# Initializing Kane's eq. of motion vector
K = [0, 0]


# Assembly Kane's Eq. of Motion
for i in range(2):
   
   K[i] = F[i] + f[i]



# -------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ SOLVING THE MOTION EQ. NUMERICALLY------------------------------------------- #

'''
The next step is to prepare the system of two equations to be solved numerically. To do so, we need to isolate the 
derivatives of u1 and u2 from the two eqs.
'''

# Isolating the higher grade derivatives from the two kane's equations
solution = sy.solve([K[0], K[1]], [u1.diff(t), u2.diff(t)])


# Extracting the two equation from the solution and converting the "symbols variables" from sympy to "python" variables 
eq = [0,0]
eq[0] = sy.lambdify((q1, u1, q2, u2, g, mp, mq, l), solution[u1.diff(t)])
eq[1] = sy.lambdify((q1, u1, q2, u2, g, mp, mq, l), solution[u2.diff(t)])


'''                    
We now write the system of differential equations and we solve it numerically.
To do so, we need to assign the necessary values to the masses, pendulum lenght...
'''

# Initial motion parameters
g = 9.81     # Gravitational acceleration [m/s^2]
mp = 1       # Mass of the cart [kg]
mq = 1       # Mass of the pendulum [kg]
l = 2        # Rod length [m]
t_max = 15   # Maximum simulation calculation time [s]


# Function that defines the system of differential equations for the numerical integration of the system
def model(y, t, mp, mq, l, g):
    q1, u1, q2, u2 = y
    u1_dot = eq[0](q1, u1, q2, u2, g, mp, mq, l)
    u2_dot = eq[1](q1, u1, q2, u2, g, mp, mq, l)
    return [u1, u1_dot, u2, u2_dot]

# Initial conditions vector for [q1_0, u1_0, q2_0, u2_0]
y0 = [0.0, 0.0, np.pi/4, 0.0]  # q1_0 = 0, u1_0 = 0, q2_0 = pi/4, u2_0 = 0
t_values = np.linspace(0, t_max, 1000)


# Numerical Integration
sol = odeint(model, y0, t_values, args=(mp, mq, l, g))

q1_values = sol[:, 0]
u1_values = sol[:, 1]
q2_values = sol[:, 2]
u2_values = sol[:, 3]




# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")



# -------------------------------------------GRAPHICS AND ANIMATION RAPPRESENTATION---------------------------------- #

# Creating the figures for the 2D animation showing how the variables change over time.
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 6))


# Graph of θ1(t)
line1, = ax1.plot([], [], label=r'$(\theta1)(t)$')
ax1.set_xlim(0, t_max)
ax1.set_ylim(np.min(q1_values*(180/np.pi)), np.max(q1_values*(180/np.pi)))
ax1.set_xlabel('Tempo (t)')
ax1.set_ylabel(r'$\theta1(t)$')
ax1.legend(loc='upper right')
ax1.grid(True)

# Graph of θ2(t)
line2, = ax2.plot([], [], label=r'$(\theta2)(t)$', color='orange')
ax2.set_xlim(0, t_max)
ax2.set_ylim(np.min(q2_values*(180/np.pi)), np.max(q2_values*(180/np.pi)))
ax2.set_xlabel('Tempo (t)')
ax2.set_ylabel(r'$\theta2(t)$')
ax2.legend(loc='upper right')
ax2.grid(True)

# Graph of θ1_dot(t)
line3, = ax3.plot([], [], label=r'$(\theta1\dot)(t)$')
ax3.set_xlim(0, t_max)
ax3.set_ylim(np.min(u1_values*(180/np.pi)), np.max(u1_values*(180/np.pi)))
ax3.set_xlabel('Tempo (t)')
ax3.set_ylabel(r'$(\theta1\dot)(t)$')
ax3.legend(loc='upper right')
ax3.grid(True)

# Graph of θ2_dot(t)
line4, = ax4.plot([], [], label=r'$(\theta2\dot)(t)$', color='orange')
ax4.set_xlim(0, t_max)
ax4.set_ylim(np.min(u2_values*(180/np.pi)-u1_values*(180/np.pi)), np.max(u2_values*(180/np.pi)))
ax4.set_xlabel('Tempo (t)')
ax4.set_ylabel(r'$\theta2\dot(t)$')
ax4.legend(loc='upper right')
ax4.grid(True)


# Function to update the animation   
def update1(frame):

    # Update graphs data
    line1.set_data(t_values[:frame], q1_values[:frame]*(180/np.pi))
    line2.set_data(t_values[:frame], q2_values[:frame]*(180/np.pi))
    line3.set_data(t_values[:frame], u1_values[:frame]*(180/np.pi))
    line4.set_data(t_values[:frame], u2_values[:frame]*(180/np.pi))

    return line1, line2, line3, line4,



# Graphs the animation for the graphs
ani_graph = FuncAnimation(fig1, update1, frames=len(t_values), interval=10, blit=True)


# ----------------------------------------DOUBLE PENDULUM MODEL ANIMATION --------------------------------------- #
fig3, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)


# Adding the line that will represent the first pendulum
pend1, = ax.plot([], [], lw=2, color='red')

# Adding the line that will represent the second pendulum
pend2, = ax.plot([], [], lw=2, color='blue')

# Adding the point P attached to the first pendulum loose extremity
point1, = ax.plot([], [], 'ro')  

# Adding the point Q attached to the second pendulum loose extremity
point2, = ax.plot([], [], 'bo')  


# Function to initialize the animation
def init():
    pend1.set_data([], [])
    pend2.set_data([], [])
    point1.set_data([], [])
    point2.set_data([], [])

    return [pend1, pend2, point1, point2]


# Update function for the animation
def update(frame):

    # The system ref frame in which the double pendulum was developed is rotated by 90° clockwise wrt to "standard" cartesian plane, so 
    # to correct this disallineament, i will apply a rotation at the pendulums extremities coordinates. 
    # More specifically   (x,y) -> (y,-x)
    # Calculating the position of the loose extremity of the first pendulum
    x_pendulum1 = l * np.cos(q1_values[frame])
    y_pendulum1 = l * np.sin(q1_values[frame])
    
    # Calcultaing the position of the loose extremity of the second pendulum
    x_pendulum2 = x_pendulum1 + l * np.cos(q2_values[frame] + q1_values[frame])
    y_pendulum2 = y_pendulum1 + l * np.sin(q2_values[frame] + q1_values[frame])

    # Updating the first pendulum position, by calculating it from the angle θ1
    pend1.set_data([0, y_pendulum1], [0, -x_pendulum1])

    # Updating the second pendulum position, by calculating it from the angle θ2
    pend2.set_data([y_pendulum1, y_pendulum2], [-x_pendulum1, -x_pendulum2])

    # Updating the points positions
    point1.set_data([y_pendulum1], [-x_pendulum1])
    point2.set_data([y_pendulum2], [-x_pendulum2])

    return [pend1, pend2, point1, point2]


# Creating the animation for the cart and the pendulm
ani_double_pendulum = FuncAnimation(fig3, update, frames=len(t_values), interval=10, blit=True, init_func=init)


# Showing the animation into a figure
plt.show()