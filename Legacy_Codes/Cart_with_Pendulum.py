# ---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI--------------------------------- #
# ---------------------------------------------SCRIPT OBJECTIVE------------------------------------------- #

# The objective of this script is to visually simulates the "Cart with Pendulum" system by numerically solving its
# equations of motion.
# The equations were derived analytically by hand and then are solved numerically by the script.
# Additionally, the script generates plots to display the values of the system's free variables.


# ----------------------------------- IMPORTING THE PACKAGES -------------------------------------------- #

# Importing the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import os

# Cleaning the terminal
os.system('cls')


# ------------------------------------- SYSTEM PARAMETERS -------------------------------------------- #

# Initial motion parameters
g = 9.81     # Gravitational acceleration [m/s^2]
m1 = 1       # Mass of the cart [kg]
m2 = 1       # Mass of the pendulum [kg]
b = 1        # Damper coefficient [N*s/m]
k = 1        # Spring constant [N/m]
l = 2        # Rod length [m]
t_max = 10   # Maximum simulation calculation time [s]


# ------------------------------------------------------------------------------------------------------- #

# Equations of motion derived manually
# x_dot_dot*(m1+m2) + b*x_dot + k*x + m2*(l/2)*(θ_dot_dot*sy.cos(θ) - ((θ_dot)**2)*sy.sin(θ)) == 0
# (2/3)*θ_dot_dot*l + x_dot_dot*sy.cos(θ) + g*sy.sin(θ) == 0

# --------------------------------------- NUMERICAL SOLUTION -------------------------------------------- #

# Function that defines the system of differential equations for the numerical integration of the system
def model(y, t, m1, m2, b, k, l, g):
    x, x_dot, θ, θ_dot = y
    x_dot_dot = (- b * x_dot - k * x + m2 * (l / 2) * (+ (θ_dot**2) * np.sin(θ) +
                 ((3*g)/(2*l) * np.sin(θ) * np.cos(θ)))) / (m1 + m2 - m2 * (3/4) * (np.cos(θ))**2)
    θ_dot_dot = (- x_dot_dot * np.cos(θ) - g * np.sin(θ)) * (3 / (2 * l))
    return [x_dot, x_dot_dot, θ_dot, θ_dot_dot]


# Initial conditions vector for [x0, x_dot0, θ0, θ_dot0]
y0 = [0.0, 0.0, np.pi/4, 0.0]  # x0 = 0, x_dot0 = 0, θ0 = pi/4, θ_dot0 = 0
t_values = np.linspace(0, t_max, 1000)


# Numerical Integration
sol = odeint(model, y0, t_values, args=(m1, m2, b, k, l, g))


# Extracting the solutions for x(t), θ(t), x_dot(t), θ_dot(t)
x_values = sol[:, 0]
x_dot_values = sol[:, 1]
θ_values = sol[:, 2]
θ_dot_values = sol[:, 3]


# ----------------------------------------- GRAPHICS AND ANIMATION RAPPRESENTATION ------------------------------------------------------ #

#  Creating the figures for the 2D animation showing how the variables change over time.
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 6))


# Graph of x(t)
line1, = ax1.plot([], [], label=r'$x(t)$')
ax1.set_xlim(0, t_max)
ax1.set_ylim(np.min(x_values)-1, np.max(x_values)+1)
ax1.set_xlabel('Tempo (t)')
ax1.set_ylabel('$x(t)$')
ax1.legend(loc='upper right')
ax1.grid(True)

# Graph of θ(t)
line2, = ax2.plot([], [], label=r'$(\theta)(t)$', color='orange')
ax2.set_xlim(0, t_max)
ax2.set_ylim(np.min(θ_values*(180/np.pi)-np.pi/4),
             np.max(θ_values*(180/np.pi))+np.pi/4)
ax2.set_xlabel('Tempo (t)')
ax2.set_ylabel(r'$\theta(t)$')
ax2.legend(loc='upper right')
ax2.grid(True)

# Graph of x_dot(t)
line3, = ax3.plot([], [], label=r'$x\dot(t)$')
ax3.set_xlim(0, t_max)
ax3.set_ylim(np.min(x_dot_values)-1, np.max(x_dot_values)+1)
ax3.set_xlabel('Tempo (t)')
ax3.set_ylabel('$x\dot(t)$')
ax3.legend(loc='upper right')
ax3.grid(True)

# Graph of θ_dot(t)
line4, = ax4.plot([], [], label=r'$(\theta\dot)(t)$', color='orange')
ax4.set_xlim(0, t_max)
ax4.set_ylim(np.min(θ_dot_values*(180/np.pi))-np.pi/4,
             np.max(θ_dot_values*(180/np.pi))+np.pi/4)
ax4.set_xlabel('Tempo (t)')
ax4.set_ylabel(r'$\theta\dot(t)$')
ax4.legend(loc='upper right')
ax4.grid(True)


# Function to update the animation
def update(frame):
    # Aggiornare i dati dei grafici
    line1.set_data(t_values[:frame], x_values[:frame])
    line2.set_data(t_values[:frame], θ_values[:frame]*(180/np.pi))
    line3.set_data(t_values[:frame], x_dot_values[:frame])
    line4.set_data(t_values[:frame], θ_dot_values[:frame]*(180/np.pi))

    return line1, line2, line3, line4,


# Graphs the animation for the graphs
ani_graph = FuncAnimation(fig1, update, frames=len(
    t_values), interval=10, blit=True)


# ------------------------------------------ CART AND PENDULUM MODELS ANIMATION ---------------------------------------------------------- #
# Creating the figure to draw the cart with pendulum animation
fig3, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)


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
    box.set_xy((x_values[frame] - box_width/2, y_values[frame] - box_height/2))

    # Calcultaing the position of the loose extremity of the pendulum
    x_pendulum = x_values[frame] + (l) * np.sin(θ_values[frame])
    y_pendulum = y_values[frame] - (l) * np.cos(θ_values[frame])

    # Updating the pendulum position
    line.set_data([x_values[frame], x_pendulum], [y_values[frame], y_pendulum])

    return [box, line]


# Creating the animation for the cart and the pendulm
ani_cart_pendulum = FuncAnimation(
    fig3, update, frames=len(t_values), interval=10, blit=True)


# Showing the animation into a figure
plt.show()
