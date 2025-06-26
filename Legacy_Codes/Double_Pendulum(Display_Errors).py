# ---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI---------------------------------#
# ---------------------------------------------SCRIPT OBJECTIVE-------------------------------------------#

# The objective of this script is to visually simulates the "Double Pendulum" system by numerically solving its
# equations of motion.
# The equations were derived analytically by hand and then are solved numerically by the script.
# Additionally, the script generates plots to display the values of the system's free variables.

# The code as now (15/12/2024) posses some displaing errors, regarding how the second pendulum is drawn on screen
# I wont investigate much on it since i already have produced newer and more advanced codes.


# ----------------------------------- IMPORTING THE PACKAGES -------------------------------------------- #

# Importing the necessary packages
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Cleaning the terminal and closing all open graphics
os.system('cls')
plt.close('all')


# ---------------------------------------- SYSTEM PARAMETERS ---------------------------------------------- #

# Initial motion parameters
g = 9.81     # Gravitational acceleration [m/s^2]
m = 1        # Mass of point P,Q [kg]
l = 2        # Rods length [m]
t_max = 10   # Maximum simulation calculation time [s]


# -------------------------------------------------------------------------------------------------------- #

# Equations of motion derived manually
# [2*u1_dot + u2_dot*cos(q2) - (u2^2)sin(q2)] = -2(g/l)*sin(q1)
# [u1_dot*cos(q2) + (u1^2)*sin(q2) + u2_dot] = -(g/l)*sin(q1+q2)

# --------------------------------------- NUMERICAL SOLUTION -------------------------------------------- #

# Function that defines the system of differential equations for the numerical integration of the system
def model(y, t, m, l, g):
    q1, u1, q2, u2 = y
    u1_dot = (-(g/l)*(2*np.sin(q1) - np.sin(q1+q2)*np.cos(q2)) + ((u2**2)
              * np.sin(q2) + (u1**2)*np.sin(q2)*np.cos(q2))/(2-np.cos(q2)**2))
    u2_dot = (-u1_dot*np.cos(q2) - (u1**2)*np.sin(q2) - (g/l)*np.sin(q1+q2))

    return [u1, u1_dot, u2, u2_dot]


# Initial conditions vector for [q1, u1, q2, u2]
y0 = [0.0, 0.0, np.pi/4, 0.0]  # q1 = 0, u1 = 0, q2 = pi/4, u2 = 0
t_values = np.linspace(0, t_max, 1000)


# Numerical Integration
sol = odeint(model, y0, t_values, args=(m, l, g))


# Extracting the solutions for q1(t), u1(t), q2(t), u2(t)
q1_values = sol[:, 0]
u1_values = sol[:, 1]
q2_values = sol[:, 2]
u2_values = sol[:, 3]


# -------------------------------------- GRAPHICS AND ANIMATION RAPPRESENTATION ----------------------------------- #

#  Creating the figures for the 2D animation showing how the variables change over time.
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
ax4.set_ylim(np.min(u2_values*(180/np.pi)), np.max(u2_values*(180/np.pi)))
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
ani_graph = FuncAnimation(fig1, update1, frames=len(
    t_values), interval=10, blit=True)


# ---------------------------------------- DOUBLE PENDULUM MODEL ANIMATION --------------------------------------- #
fig3, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)


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
ani_double_pendulum = FuncAnimation(fig3, update, frames=len(
    t_values), interval=10, blit=True, init_func=init)


# Showing the animation into a figure
plt.show()
