# ---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI----------------------------------------------- #
# ---------------------------------------------SCRIPT OBJECTIVE--------------------------------------------------------- #

# The objective of this script is to derive the eq. of motion of the "Cart with Pendulum" system by 
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
t = sy.symbols('t')                # Definition of the time variable
g = sy.symbols('g')                # Gravitational acceleration [m/s^2]
m1 = sy.symbols('m1')              # Mass of point P [kg]
m2 = sy.symbols('m2')              # Mass of point Q [kg]
b = sy.symbols('b')                # Damper coefficient [N*s/m]
k = sy.symbols('k')                # syring constant [N/m]
l = sy.symbols('l')                # Rod length [m]
h = sy.symbols('h')                # Box height [m]
w = sy.symbols('h')                # Box whidt [m]
a = sy.symbols('a')                # Cart center of mass's height wrt x axis [m] 
t_max = 10                         # Maximum simulation calculation time [s]


# Inertia of the cart, along z axis [kg*m^2].
I_m1 = sy.Matrix([[0,0,0],[0,0,0],[0,0,(1/12)*m1*(h**2 + w**2)]])     

# Inertia of the pendulum, along z axis [kg*m^2]. 
I_m2 = sy.Matrix([[0,0,0],[0,0,0],[0,0,(1/12)*m2*(l**2)]])           


# Coordinates definition
q1 = me.dynamicsymbols('q1')        # Linear displacement of the cart center of mass (x coordinate), function of time "t"
q2 = me.dynamicsymbols('q2')        # Angular displacement of the pendulum rod (q2 coordinate), function of time "t"



# -----------------------------------------REFERENCE FRAME DEFINITION-------------------------------------------------- #
'''
Let's start by defining an inertial reference frame N, and a n° of additional ref. frame equal to the n° of bodies our
system has.
'''
# Reference Frame definition
N = me.ReferenceFrame('N')       # Inertial ref. Frame, fixed with the pavement   
C = me.ReferenceFrame('C')       # Cart body ref. Frame
P = me.ReferenceFrame('P')       # Pendulum body ref. Frame, with origin centered on the c.o.m. of the pendulum (G)


'''
Now we define the origin of our reference frames. Since the body frames will be centered of the bodies center of mass,
the reference frame's origin point will coincide with their c.o.m.
We need to refer the ref. frame one to the other, more syecifically, defining the position vectors from point O->A 
and O->G, and also defining the relative orientation between the ref. Frame C and P wrt the inertial one N.
If not syecified, the position of every point we define are centered on the coordinates (0,0,0), the origin of our
ref. system.
'''
# Reference Frame's origin definition
O = me.Point('O')               # Inertial ref. Frame origin
A = me.Point('A')               # Cart body ref. Frame origin, centered on the c.o.m. of the cart (A)
G = me.Point('G')               # Pendulum body ref. Frame origin, centered on the c.o.m. of the pendulum (G)


# Setting the relative position between the frame's origins
A.set_pos(O, q1*N.x + a*N.y)                                               #Setting the center of mass of the cart relative to the inertial ref. Frame
A.set_pos(A, 0*C.x + 0*C.y + 0*C.z)                                        #Setting the center of mass of the cart as the origin of its own body frame
 
G.set_pos(A, (l/2)*sy.sin(q2)*C.x + (- (l/2)*sy.cos(q2))*C.y)              #Setting the center of mass of the prendulum relative to the cart ref. Frame
G.set_pos(G, 0*P.x + 0*P.y + 0*P.z)                                        #Setting the center of mass of the pendulum as the origin of its own body frame


# Setting the relative orientation between the frames
C.orient_axis(N, 0, N.z)                                 #Defining the cart ref. frame as rotated of 0 wrt the Inertial. By doing so, i tell the code that the two frames have their axes alligned
P.orient_axis(C, -(np.pi/2 - q2), C.z)                   #Defining the pendulum ref. frame as rotated of -(pi/2 - q2) wrt to the cart ref. frame


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
x_A = A.pos_from(O)
x_G = G.pos_from(O)


# Velocity vector definition
v_A = A.vel(N)
v_G = G.vel(N)


# Angular velocity vector
omega_C = C.ang_vel_in(N)
omega_P = P.ang_vel_in(N)


# Total acceleration vector definition
a_A = A.acc(N)
a_G = G.acc(N)


# Total angular acceleration vector definition
omegadot_C = C.ang_acc_in(N) 
omegadot_P = P.ang_acc_in(N)



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
ui = [v_A.dot(N.x), v_A.dot(N.y), v_A.dot(N.z), v_G.dot(N.x), v_G.dot(N.y), v_G.dot(N.z)]
omega_i = [omega_C.dot(C.x), omega_C.dot(C.y), omega_C.dot(C.z), omega_P.dot(P.x), omega_P.dot(P.y), omega_P.dot(P.z)]


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
v_A = v_A.subs(substitutions)
v_G = v_G.subs(substitutions)
a_A = a_A.subs(substitutions)
a_G = a_G.subs(substitutions)
omega_C = omega_C.subs(substitutions)
omega_P = omega_P.subs(substitutions)
omegadot_C = omegadot_C.subs(substitutions)
omegadot_P = omegadot_P.subs(substitutions)


# ------------------------------------------ PARTIAL VELOCITIES CALCULATIONS--------------------------------------------------- #
'''
Next step is to define the partial velocities for our system. To do so, we need to get the terms that multiply the
generalized velocities inside total velocities vectors. we know that this terms are linearly multypling the generalized
speeds, so we can simply partially derive the total velocity vectors for the gen. speeds to get these terms.
'''
# Partial Velocities for the system
v_A_part1 = v_A.diff(u1, N)
v_A_part2 = v_A.diff(u2, N)                                                
v_G_part1 = v_G.diff(u1, N)
v_G_part2 = v_G.diff(u2, N)                              
omega_C_part1 = omega_C.diff(u1, N)
omega_C_part2 = omega_C.diff(u2, N)
omega_G_part1 = omega_P.diff(u1, N)
omega_G_part2 = omega_P.diff(u2, N)


# Rearranging the partial velocities into matrices for ease of use
v_A_part = sy.Matrix([[v_A_part1.dot(N.x), v_A_part1.dot(N.y), v_A_part1.dot(N.z)],[v_A_part2.dot(N.x), v_A_part2.dot(N.y), v_A_part2.dot(N.z)]])
v_G_part = sy.Matrix([[v_G_part1.dot(N.x), v_G_part1.dot(N.y), v_G_part1.dot(N.z)],[v_G_part2.dot(N.x), v_G_part2.dot(N.y), v_G_part2.dot(N.z)]])
omega_C_part = sy.Matrix([[omega_C_part1.dot(N.x), omega_C_part1.dot(N.y), omega_C_part1.dot(N.z)],[omega_C_part2.dot(N.x), omega_C_part2.dot(N.y), omega_C_part2.dot(N.z)]])
omega_G_part = sy.Matrix([[omega_G_part1.dot(N.x), omega_G_part1.dot(N.y), omega_G_part1.dot(N.z)],[omega_G_part2.dot(N.x), omega_G_part2.dot(N.y), omega_G_part2.dot(N.z)]])



# ----------------------------------------- FORCES & TORQUES ACTING ON OUR BODIES---------------------------------------------- #
'''
We then define the External forces acting on our system, referred to the center of masses of the two bodies and 
projected along the inertial reference system.
Pay attention to the fact that every vector from this line on will be represented as a Matrix [1xn], that beacause
we can use the sympy toolbox to make us able to derive and manage the vectors like we are able to do with pen and papers
'''
# Forces & Torques acting on body 1 (Cart)
Fp_1 = sy.Matrix([0, -m1*g, 0])
Fel = sy.Matrix([-k*q1, 0, 0])
Fdamp = sy.Matrix([-b*u1, 0, 0])

Torq_1 = sy.Matrix([0, 0, 0])


# Forces & Torques acting on body 2 (Pendulum)
Fp_2 = sy.Matrix([0, -m2*g, 0])

Torq_2 = sy.Matrix([0, 0, 0])


# Calculating the resultant forces & torques acting on the two bodies
Ftot_1 = Fp_1 + Fel + Fdamp      #Total force vector acting on the cart
Torqtot_1 = Torq_1               #Total torque vector acting on the cart

Ftot_2 = Fp_2                    #Total force vector acting on the pendulum
Torqtot_2 = Torq_2               #Total torque vector acting on the pendulum



# -------------------------------------------- GENERALIZED INERTIA & ACTIVE FORCES--------------------------------------------- #

# Rearranging quantities into matrices for ease of use
# Angular velocity matrix for the entire system
omega = sy.Matrix([[omega_C.dot(N.x), omega_C.dot(N.y), omega_C.dot(N.z)],[omega_P.dot(N.x), omega_P.dot(N.y), omega_P.dot(N.z)]])    

# Acceleration matrix for the entire system
a = sy.Matrix([[a_A.dot(N.x), a_A.dot(N.y), a_A.dot(N.z)],[a_G.dot(N.x), a_G.dot(N.y), a_G.dot(N.z)]])    

# Angular acceleration matrix for the entire system
omegadot = sy.Matrix([[omegadot_C.dot(N.x), omegadot_C.dot(N.y), omegadot_C.dot(N.z)],[omegadot_P.dot(N.x), omegadot_P.dot(N.y), omegadot_P.dot(N.z)]])

# Mass matrix for the entire system
M = sy.Matrix([m1, m2])

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
   F[i] = - (M[0] * a[0,0:3].dot(v_A_part[i,0:3]) + ((omegadot[0,0:3]*I_m1) + omega[0,0:3].cross(I_m1*(omega[0,0:3].T))).dot(omega_C_part[i,0:3])) - (M[1] * a[1,0:3].dot(v_G_part[i,0:3]) + ((omegadot[1,0:3]*I_m2) + omega[1,0:3].cross(I_m2*(omega[1,0:3].T))).dot(omega_G_part[i,0:3]))

   # Generalized Active forces
   f[i] = (M[0] * Ftot[0,0:3].dot(v_A_part[i,0:3]) + (Torqtot[0,0:3]).dot(omega_C_part[i,0:3])) + (M[1] * Ftot[1,0:3].dot(v_G_part[i,0:3]) + (Torqtot[1,0:3]).dot(omega_C_part[i,0:3]))



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
eq[0] = sy.lambdify((q1, u1, q2, u2, g, m1, m2, b, k, l), solution[u1.diff(t)])
eq[1] = sy.lambdify((q1, u1, q2, u2, g, m1, m2, b, k, l), solution[u2.diff(t)])


'''
We now write the system of differential equations and we solve it numerically.
To do so, we need to assign the necessary values to the masses, pendulum lenght...
'''

# Initial motion parameters
g = 9.81     # Gravitational acceleration [m/s^2]
m1 = 1       # Mass of the cart [kg]
m2 = 1       # Mass of the pendulum [kg]
b = 1        # Damper coefficient [N*s/m]
k = 1        # Spring constant [N/m]
l = 2        # Rod length [m]
t_max = 10   # Maximum simulation calculation time [s]


# Function that defines the system of differential equations for the numerical integration of the system
def model(y, t, m1, m2, b, k, l, g):
    q1, u1, q2, u2 = y
    u1_dot = eq[0](q1, u1, q2, u2, g, m1, m2, b, k, l)
    u2_dot = eq[1](q1, u1, q2, u2, g, m1, m2, b, k, l)
    return [u1, u1_dot, u2, u2_dot]


# Initial conditions vector for [q1_0, u1_0, q2_0, u2_0]
y0 = [0.0, 0.0, np.pi/4, 0.0]  # q1_0 = 0, u1_0 = 0, q2_0 = pi/4, u2_0 = 0
t_values = np.linspace(0, t_max, 1000)


# Numerical Integration
sol = odeint(model, y0, t_values, args=(m1, m2, b, k, l, g))


# Extracting the solutions for q1(t), q2(t), u1(t), u2(t) 
q1_values = sol[:, 0]
u1_values = sol[:, 1]
q2_values = sol[:, 2]
u2_values = sol[:, 3]


# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

end = time.perf_counter()

# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")



# ---------------------------------------------GRAPHICS AND ANIMATION -------------------------------------------- #

#  Creating the figures for the 2D animation showing how the variables change over time.
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 6))


# Graph of x(t)
line1, = ax1.plot([], [], label=r'$x(t)$')
ax1.set_xlim(0, t_max)
ax1.set_ylim(np.min(q1_values), np.max(q1_values))
ax1.set_xlabel('Tempo (t)')
ax1.set_ylabel('$x(t)$')
ax1.legend(loc='upper right')
ax1.grid(True)

# Graph of θ(t)
line2, = ax2.plot([], [], label=r'$(\theta)(t)$', color='orange')
ax2.set_xlim(0, t_max)
ax2.set_ylim(np.min(q2_values*(180/np.pi)), np.max(q2_values*(180/np.pi)))
ax2.set_xlabel('Tempo (t)')
ax2.set_ylabel(r'$\theta(t)$')
ax2.legend(loc='upper right')
ax2.grid(True)

# Graph of x_dot(t)
line3, = ax3.plot([], [], label=r'$x\dot(t)$')
ax3.set_xlim(0, t_max)
ax3.set_ylim(np.min(u1_values), np.max(u1_values))
ax3.set_xlabel('Tempo (t)')
ax3.set_ylabel('$x\dot(t)$')
ax3.legend(loc='upper right')
ax3.grid(True)

# Graph of θ_dot(t)
line4, = ax4.plot([], [], label=r'$(\theta\dot)(t)$', color='orange')
ax4.set_xlim(0, t_max)
ax4.set_ylim(np.min(u2_values*(180/np.pi)), np.max(u2_values*(180/np.pi)))
ax4.set_xlabel('Tempo (t)')
ax4.set_ylabel(r'$\theta\dot(t)$')
ax4.legend(loc='upper right')
ax4.grid(True)


# Function to update the animation   
def update(frame):
    # Aggiornare i dati dei grafici
    line1.set_data(t_values[:frame], q1_values[:frame])
    line2.set_data(t_values[:frame], q2_values[:frame]*(180/np.pi))
    line3.set_data(t_values[:frame], u1_values[:frame])
    line4.set_data(t_values[:frame], u2_values[:frame]*(180/np.pi))

    return line1, line2, line3, line4,


# Graphs the animation for the graphs
ani_graph = FuncAnimation(fig1, update, frames=len(t_values), interval=10, blit=True)



# ----------------------------------------CART AND PENDULUM MODELS ANIMATION ------------------------------------------------- #

# Creating the figure to draw the cart with pendulum animation
fig3, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)


# Creation of the box that rappresent the cart
# Here the box will have a 1x1 dimension, and its position will be definited by defining the coordinates of its bottom left corner
box_width = 1
box_height = 1
box = plt.Rectangle((0, 0), box_width, box_height, color='blue')


# Definition of the y-axis values, necessary to describe the motion of the cart. These are all zero because the cart moves only along the x-axis.
y_values = np.linspace(box_height/2, box_height/2, len(t_values))


# Adding the line that will represent the pendulum
line, = ax.plot([], [], lw=3, color='red')


# Adding the box into the figure
ax.add_patch(box)


# Function to initialize the animation
def init():
    box.set_xy((0, 0))
    line.set_data([], [])
    return [box, line]


# Update function for the animation
def update(frame):
    # Updating the box position by passing the coordinates of its center of mass through x(t) and y(t)
    box.set_xy((q1_values[frame] - box_width/2, y_values[frame] - box_height/2))

    # Calcultaing the position of the loose extremity of the pendulum
    x_pendulum = q1_values[frame] + (l) * np.sin(q2_values[frame])  
    y_pendulum = y_values[frame] - (l) * np.cos(q2_values[frame])

    # Updating the pendulum position
    line.set_data([q1_values[frame], x_pendulum], [y_values[frame], y_pendulum])

    return [box, line]


# Creating the animation for the cart and the pendulm
ani_cart_pendulum = FuncAnimation(fig3, update, frames=len(t_values), interval=10, blit=True)


# Showing the animation into a figure
plt.show()

