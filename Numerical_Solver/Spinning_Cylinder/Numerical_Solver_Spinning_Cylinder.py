#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI-----------------------------------------------#
#------------------------------------------- SCRIPT OBJECTIVE ---------------------------------------------------------#

# The objective of this script is to solve the eq. of motion of a system by solving numerically the linear matrix system COEF = udot*RHS




#-------------------------------------------- IMPORTING THE NECESSARY PACKAGES ------------------------------------------#

import numpy as np
import sympy as sy
import sympy.physics.mechanics as me
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import json
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

#Cleaning the terminal 
os.system('cls')


# ---------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

start = time.perf_counter()


#--------------------------------------------------- LOADING THE MATRICES ----------------------------------------------#

# Reading the Json file
with open('Kane_MatrixForm_SpinningCylinder\Dynamical_Matrices_SpinningCylinder.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)

with open('Kane_MatrixForm_SpinningCylinder\Kinematical_Matrices_SpinningCylinder.json', 'r') as json_file:
    data_Mat_Cin = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']['COEF_Dyn']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']['RHS_Dyn']))

# Reconstructing the Kinematical matrices
COEF_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['COEF_Kin']['COEF_Kin']))
RHS_Kin_loaded = sy.Matrix(sy.sympify(data_Mat_Cin['RHS_Kin']['RHS_Kin']))


# --------------------------------------------------------------- #

# Initializing the variables
t = sy.symbols('t')                # Definition of the time variable
g = sy.symbols('g')                # Gravitational acceleration [m/s^2]
m0 = sy.symbols('m0')              # Mass of body 1 [kg]
m1 = sy.symbols('m1')              # Mass of body 2 [kg]
b = sy.symbols('b')                # Damper coefficient [N*s/m]
k = sy.symbols('k')                # spring constant [N/m]
r = sy.symbols('l')                # Rod length [m]
h = sy.symbols('h')                # Box height [m]
w = sy.symbols('w')                # Box whidt [m]
a = sy.symbols('a')                # Cart center of mass's height wrt x axis [m] 
T1, T2, T3 = sy.symbols('T1 T2 T3')                # Torque applied


# Initializing the gen.coordinates and speeds
q1,q2,q3,q4,q5,q6 = sy.symbols('q1 q2 q3 q4 q5 q6')
u1,u2,u3,u4,u5,u6  = sy.symbols('u1 u2 u3 u4 u5 u6')

#----------------------------------------- REFERENCE FRAME DEFINITION --------------------------------------------------#

# Inertial Reference Frame definition
n = me.ReferenceFrame('n')          # Inertial ref. Frame, fixed with the pavement   

# Bodies Ref. Frame definitions
b0 = me.ReferenceFrame('b0')        # Cart body ref. Frame, with origin centered on point A 



# Inertial Reference Frame origin definition
O = me.Point('O')                   # Inertial ref. Frame origin

# Bodies Ref. Frame origin definition
B0 = me.Point('B0')                 # Cart ref. Frame origin




# Setting the relative position between the frame's origins
B0.set_pos(O, q4*n.x + q5*n.y + q6*n.z)                      # Setting point B0 relative to the inertial ref. Frame
B0.set_pos(B0, 0*b0.x + 0*b0.y + 0*b0.z)                     # Setting point B0 as the origin of the cart ref. frame




# --------------------------------------------------------------------------- #

# Setting the relative orientation between the frames

# Defining the cart ref. frame as rotated wrt the Inertial
b0.orient_body_fixed(n, (q1, q2, q3), 'zxy')

# Setting the velocities of the root body
B0.set_vel(n, u4*n.x + u5*n.y + u6*n.z)
b0.set_ang_vel(n, u1*b0.x + u2*b0.y + u3*b0.z)

# Setting the velocity of the inertial ref. frame to 0, to allow the calculations to be developed
O.set_vel(n, 0)
n.set_ang_vel(n, 0)


# Putting the body ref. Frame inside a list, so i can express vector quantities easier
ref_body = [b0]
# Inertia Dyadic for each body, centered in their respective com and expressed along their principal inertia axis
I_0 = ((1/12)*m0*(3*r**2 + h**2)*(me.outer(b0.x, b0.x) + me.outer(b0.y, b0.y)) + 0.5*(m0*r**2)*(me.outer(b0.z,b0.z))) #+ 0.5*(me.outer(b0.z,b0.x) + me.outer(b0.x,b0.z))).to_matrix(b0) 


# ----------------------------------------- SOLVING THE SYSTEM ---------------------------------------------------- #

# Now we have to create a dictionary for the symbolic variables substituition inside the matrices. Practically we need
# to specify the masses, lenghts, ..., values for the system parameters

# Dictionary
substitution = {
    m0: 5,    # Massa del carrello [kg]
    h: 6,     # Costante della molla [N/m]
    r: 0.5,     # Lunghezza della sbarra [m]
}

# Substituing the symbolic variables with their values inside the matrices
# Dynamical Matrices
COEF_Dyn_loaded = COEF_Dyn_loaded.xreplace(substitution)
RHS_Dyn_loaded = RHS_Dyn_loaded.xreplace(substitution)

#--------------------------------------------------- CALCULATIONS -------------------------------------------------------#

# We assemble the giant mass matrix for the system we want to solve, its composed by the mass matrix of the dynamical and 
# kinematicl equations. The same we do to assemble the giant forcing vector
Mass_Matrix = sy.BlockDiagMatrix(COEF_Dyn_loaded, COEF_Kin_loaded)
Forcing_Vector = sy.Matrix.vstack(RHS_Dyn_loaded, RHS_Kin_loaded)


# Transforming the matrices from "blockmatrices" to "normal matrices"
Mass_Matrix_dense = Mass_Matrix.as_mutable()
Forcing_Vector_dense = Forcing_Vector.as_mutable()

# Translating the symbolic matrices from sympy to numpy
all_symbolic_var = (u1,u2,u3,u4,u5,u6,q1,q2,q3,q4,q5,q6,T1,T2,T3)
Mass_Matrix_lambdified = sy.lambdify(all_symbolic_var, Mass_Matrix_dense, 'numpy')
Forcing_Vector_lambdified = sy.lambdify(all_symbolic_var, Forcing_Vector_dense,'numpy')


def system(t, State_Vector):

    # Unpacking
    u1 = State_Vector[0]
    u2 = State_Vector[1]
    u3 = State_Vector[2]
    u4 = State_Vector[3]
    u5 = State_Vector[4]
    u6 = State_Vector[5]

    q1 = State_Vector[6]
    q2 = State_Vector[7]
    q3 = State_Vector[8]
    q4 = State_Vector[9]
    q5 = State_Vector[10]
    q6 = State_Vector[11]

    t = t


   # ------------------------------ #
   # Force and Torques timed application
    
    # Torque applied for 1s<t<2.5s
    if t > 1 and t < 2.0:
        
        T1 = 0
        T2 = 0
        T3 = 0

    else:
        T1 = 0
        T2 = 0
        T3 = 0

    # Sostituisci le variabili simboliche con i valori numerici in Mass_Matrix e Forcing_Vector
    Mass_Matrix_eval = Mass_Matrix_lambdified(u1,u2,u3,u4,u5,u6,q1,q2,q3,q4,q5,q6,T1,T2,T3)
    Forcing_Vector_eval = Forcing_Vector_lambdified(u1,u2,u3,u4,u5,u6,q1,q2,q3,q4,q5,q6,T1,T2,T3)
   
    try:
        
        # Risolvere il sistema lineare senza calcolare esplicitamente l'inversa
        State_Vector_dot = np.linalg.solve(Mass_Matrix_eval, Forcing_Vector_eval)
        
    except np.linalg.LinAlgError:
        
        # Se la matrice è singolare, puoi usare la pseudo-inversa
        State_Vector_dot = np.dot(np.linalg.pinv(Mass_Matrix_eval), Forcing_Vector_eval)
      
    # Restituire le derivate di tutte le variabili
    print(t)

    return State_Vector_dot.flatten()

# Condizioni iniziali
      #u1, u2, u3, u4, u5, u6, q1, q2, q3, q4, q5, q6 
y0 = [2.0,0.0,1.0,0.1,0.3,0.4,0.0,0.0,2.0,0.0,0.0,0.0]


# Intervallo di tempo per la soluzione
t_span = (0.0, 15)  # Risolvi dal tempo 0 al tempo 10
t_eval = np.linspace(t_span[0], t_span[1], 1500)  # 1000 punti di valutazione

solution = solve_ivp(system, t_span, y0, t_eval=t_eval, vectorized=False, method='RK23', rtol=1e-4, atol=1e-5, dense_output=True, max_step = 0.3)


# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

end = time.perf_counter()



# ------------------------------------------------ GRAFICARE LA SOLUZIONE -------------------------------------------- #

# Estrazione dei risultati
u1_solution = solution.y[0]
u2_solution = solution.y[1]
u3_solution = solution.y[2]
u4_solution = solution.y[3]
u5_solution = solution.y[4]
u6_solution = solution.y[5]
q1_solution = solution.y[6]
q2_solution = solution.y[7]
q3_solution = solution.y[8]
q4_solution = solution.y[9]
q5_solution = solution.y[10]
q6_solution = solution.y[11]



# Grafico della soluzione
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(solution.t, u1_solution*(180/np.pi), label="omegaz", color='r')
plt.xlabel("Tempo")
plt.ylabel("u1 (velocità angolare)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(solution.t, u2_solution*(180/np.pi), label="omegax", color='g')
plt.xlabel("Tempo")
plt.ylabel("u2 (velocità angolare)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(solution.t, u3_solution*(180/np.pi), label="omegay", color='b')
plt.xlabel("Tempo")
plt.ylabel("u3 (velocità angolare)")
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(solution.t, u4_solution, label="vx" ,color='r')
plt.xlabel("Tempo")
plt.ylabel("u4 (velocità lineare)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(solution.t, u5_solution, label="vy", color='g')
plt.xlabel("Tempo")
plt.ylabel("u5 (velocità lineare)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(solution.t, u6_solution, label="vz", color='b')
plt.xlabel("Tempo")
plt.ylabel("u6 (velocità lineare)")
plt.grid(True)
plt.legend()

plt.tight_layout()


# ----------------------------------------------------------------------------------------------- #

# Defining the variables
Etot = [0]*len(solution.t)
Ecin = [0]*len(solution.t)
Erot = [0]*len(solution.t)

# Angular momentum calculations
L_components = [0]*len(solution.t)
L_mag = [0]*len(solution.t)
w_cog = [0]*len(solution.t)
Itot = sy.lambdify((q1,q2,q3),(I_0.to_matrix(b0).xreplace(substitution)))

for i in range(len(solution.t)):
    w_cog[i] = np.array([u2_solution[i], u3_solution[i], u1_solution[i]])
    v_cog = np.array([u4_solution[i], u5_solution[i], u6_solution[i]]) 
    L_components[i] =  (Itot(q1_solution[i],q2_solution[i],q3_solution[i]) @ w_cog[i]) #- np.cross(-baricenter_loaded_lambdified_eval[i].flatten(), (Msat*v_cog))
    L_mag[i] = (L_components[i][0]**2 + L_components[i][1]**2 + L_components[i][2]**2)**0.5



    # Calculating the system linear and rotational kinetic energy
    Ecin[i] = 0.5*5*(np.linalg.norm(v_cog))**2
    Erot[i] =  0.5*w_cog[i].T @ Itot(q1_solution[i],q2_solution[i],q3_solution[i]) @ w_cog[i]
    
    
    # Calculating the system potential energy
    Epot = 0

    # Calculating the total energy at each time step
    Etot[i] = Ecin[i] + Erot[i] + Epot


# Converting L_components from a list to an array
L_components = np.array(L_components)

# ------------------------------------------------------------------ #
# Momentum conservation

plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(solution.t, L_components[:,0], label="Angular momentum along x",color='r')
plt.xlabel("t")
plt.ylabel("Lx")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(solution.t, L_components[:,1], label="Angular momentum along y", color='g')
plt.xlabel("t")
plt.ylabel("Ly")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(solution.t, L_components[:,2], label="Angular momentum along z", color='b')
plt.xlabel("t")
plt.ylabel("Lz")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(solution.t, L_mag, label="Angular momentum vector magnitude", color='k')
plt.xlabel("t")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()

plt.tight_layout()


# ------------------------------------------------------------------ #
# Energy conservation

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(solution.t, Ecin, label="Kinetic energy magnitude", color='r')
plt.xlabel("t")
plt.ylabel("Value")
plt.grid(True)
plt.legend()


plt.subplot(3, 1, 2)
plt.plot(solution.t, Erot, label="Rotational energy magnitude", color='b')
plt.xlabel("t")
plt.ylabel("Value")
plt.grid(True)
plt.legend()


plt.subplot(3, 1, 3)
plt.plot(solution.t, Etot, label="Total energy magnitude", color='k')
plt.xlabel("t")
plt.ylabel("Value")
plt.grid(True)
plt.legend()

plt.tight_layout()



# -------------------------------------------- ANIMAZIONE CILINDRO ----------------------------------------- #

# Creazione della figura
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
time_text = fig.text(0.1, 0.9, '')

# Lunghezza del cilindro
height = 4.0

# Funzione per aggiornare l'animazione
def update(frame):
    ax.clear()  # Pulisce il grafico precedente

    # Centro dell'asse del cilindro
    center = np.array([q4_solution[frame], q5_solution[frame], q6_solution[frame]])

    # Disegna il centro come un punto
    ax.scatter(*center, color='red', s=50, label="Centro Asse")


    ax.set_title('x axis: red, y axis: blue, z axis: green')
    time_text.set_text(f"t = {solution.t[frame]:.2f} s")


    # Inertial ref frame
    x_n = O.pos_from(O).to_matrix(n)[0].xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    y_n = O.pos_from(O).to_matrix(n)[1].xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    z_n = O.pos_from(O).to_matrix(n)[2].xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    xv_n = n.x.to_matrix(n).xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    yv_n = n.y.to_matrix(n).xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    zv_n = n.z.to_matrix(n).xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})


    # Root body rootated ref frame
    x_B0 = B0.pos_from(O).to_matrix(n)[0].xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    y_B0 = B0.pos_from(O).to_matrix(n)[1].xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    z_B0 = B0.pos_from(O).to_matrix(n)[2].xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    xv_B0 = b0.x.to_matrix(n).xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    yv_B0 = b0.y.to_matrix(n).xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})
    zv_B0 = b0.z.to_matrix(n).xreplace(substitution).xreplace({q1:q1_solution[frame], q2:q2_solution[frame], q3:q3_solution[frame],q4:q4_solution[frame],q5:q5_solution[frame],q6:q6_solution[frame]})

    # Inertial ref frame
    ax.quiver(x_n, y_n, z_n, xv_n[0], xv_n[1], xv_n[2], color='red', length = 2 ,normalize=True)
    ax.quiver(x_n, y_n, z_n, yv_n[0], yv_n[1], yv_n[2], color='blue', length = 2 ,normalize=True)
    ax.quiver(x_n, y_n, z_n, zv_n[0], zv_n[1], zv_n[2], color='green', length = 2 ,normalize=True)

    # Root body rotated ref frame
    ax.quiver(x_B0, y_B0, z_B0, xv_B0[0], xv_B0[1], xv_B0[2], color='red', length = 1.5 ,normalize=True)
    ax.quiver(x_B0, y_B0, z_B0, yv_B0[0], yv_B0[1], yv_B0[2], color='blue', length = 1.5 ,normalize=True)
    ax.quiver(x_B0, y_B0, z_B0, zv_B0[0], zv_B0[1], zv_B0[2], color='green', length = 1.5 ,normalize=True)

    # Root body rotated ref frame
    ax.quiver(x_B0, y_B0, z_B0, w_cog[frame][0], w_cog[frame][1], w_cog[frame][2], color='black', length = 2 ,normalize=True)


    # Imposta i limiti degli assi
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    # Normalizza il vettore zv_B0 e scala in base all'altezza del cilindro
    zv_B0_numeric = np.array([zv_B0[0], zv_B0[1], zv_B0[2]], dtype=float)  # Converti in NumPy array
    zv_B0_unit = zv_B0_numeric / np.linalg.norm(zv_B0_numeric)  # Ora puoi normalizzarlo
    axis_start = center - (height / 2) * zv_B0_unit  # Estremo inferiore
    axis_end = center + (height / 2) * zv_B0_unit  # Estremo superiore

    # Disegna il nuovo asse del cilindro allineato con zv_B0
    ax.plot([axis_start[0], axis_end[0]], 
            [axis_start[1], axis_end[1]], 
            [axis_start[2], axis_end[2]], 
            color='black', lw=5, label="Asse Cilindro")




# Crea l'animazione
ani = FuncAnimation(fig, update, frames=len(solution.t), interval=50)




plt.show()



# Showing the computation time
print(f"The calculations required time was: {end - start:.4f} seconds")

print()
print('Codice terminato')

