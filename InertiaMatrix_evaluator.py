#------------------------------------ IMPORTING THE NECESSARY PACKAGES -------------------------------------------------#

import sympy as sy
import numpy as np
import sympy.physics.mechanics as me
import os
import json
import time


# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

#Cleaning the terminal 
os.system('cls')



# ---------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #

start = time.perf_counter()



#------------------------------------------ SYMBOLIC VARIABLES ----------------------------------------------------#    
print('Loading symbols, generalized coordinates and speeds...')

# Initial motion parameters
t = sy.symbols('t')                 # Definition of the time variable
l = sy.symbols('l')                 # Spacecraft bus length [m]
w = sy.symbols('w')                 # Spacecraft bus width [m]
h = sy.symbols('h')                 # Spacecraft bus height [m]
d = sy.symbols('d')                 # Spacecraft bus-arm joint distance [m]
dg = sy.symbols('dg')               # Spacecraft bus joint distance [m]
bl = sy.symbols('bl')               # Robotic arm long piece length [m]
hl = sy.symbols('hl')               # Robotic arm long piece height [m]
wl = sy.symbols('wl')               # Robotic arm long piece width [m]
bs = sy.symbols('bs')               # Robotic arm short piece length [m]
hs = sy.symbols('hs')               # Robotic arm short piece height [m]
ws = sy.symbols('ws')               # Robotic arm short piece width [m]
mlongarm = sy.symbols('mlongarm')   # Robotic Long arm piece mass [kg]
mshortarm = sy.symbols('mshortarm') # Robotic Short arm piece mass [kg]
mbus = sy.symbols('mbus')           # Spacecraft bus mass [kg]


# Generalized Coordinates definition
q1, q2, q3, q4, q7, q10, q13, q18, q21, q24, q27, q28, q31, q34, q37, q42, q45, q48, q51, q52, q53, q54 = sy.symbols('q1 q2 q3 q4 q7 q10 q13 q18 q21 q24 q27 q28 q31 q34 q37 q42 q45 q48 q51 q52 q53 q54')



#--------------------------------------------------- LOADING THE FILES ----------------------------------------------#
print('Loading files...')

# Reading the Json Inertia file
with open('Itot_baricentric_satellite.json', 'r') as json_file:
    data_Inertia = json.load(json_file)

# Reading the Json solver file
with open('solution_4arms_Preloadedsprings_dampers_FINAL.json', 'r') as json_file:
    data_data = json.load(json_file)



# -------------------------- EVALUATING THE INERTIA MATRIX WITH THE SOLUTIONS DATA---------------------------------- #

# Reconstructing the Central Inertia matrix
Itot_baricentric_satellite = sy.Matrix(sy.sympify(data_Inertia['Itot_baricentric_satellite']['Itot_baricentric_satellite']))

# Reconstructing the rotation matrix
Cn_b00 = sy.Matrix(sy.sympify(data_Inertia['Cn_b00']['Cn_b00']))


# ------------------------------------------------------------------ #
# Extracting the solver data from the JSON

# Extracting time
t_loaded = np.array(data_data["t"])  

# Extracting the bus angular position
q1_data = np.array(data_data['y'][22])
q2_data = np.array(data_data['y'][23])
q3_data = np.array(data_data['y'][24])

# Robotic arm 1
q4_data = np.array(data_data['y'][25])
q7_data = np.array(data_data['y'][26])
q10_data = np.array(data_data['y'][27])
q13_data = np.array(data_data['y'][28])

# Robotic arm 2
q18_data = np.array(data_data['y'][29])
q21_data = np.array(data_data['y'][30])
q24_data = np.array(data_data['y'][31])
q27_data = np.array(data_data['y'][32])

# Robotic arm 3
q28_data = np.array(data_data['y'][33])
q31_data = np.array(data_data['y'][34])
q34_data = np.array(data_data['y'][35])
q37_data = np.array(data_data['y'][36])

# Robotic arm 4
q42_data = np.array(data_data['y'][37])
q45_data = np.array(data_data['y'][38])
q48_data = np.array(data_data['y'][39])
q51_data = np.array(data_data['y'][40])

# Extracting the bus linear position
q52_data = np.array(data_data['y'][41])
q53_data = np.array(data_data['y'][42])
q54_data = np.array(data_data['y'][43])



# ------------------------------------------ SUBSTITUING THE CONSTANT VALUES  ------------------------------------ #

# Dictionary
substitution = {
    l: 0.6,                 # Spacecraft bus length [m]
    w: 0.5,                 # Spacecraft bus width [m]
    h: 0.5,                 # Spacecraft bus height [m]
    d: 0.1,                 # Spacecraft bus-arm joint distance [m]
    dg: 0.025,              # Spacecraft bus joint distance [m]
    bl: 0.25,               # Robotic arm long piece length [m]
    hl: 0.05,               # Robotic arm long piece height [m]
    wl: 0.05,               # Robotic arm long piece width [m]
    bs: 0.16,               # Robotic arm short piece length [m]
    hs: 0.05,               # Robotic arm short piece height [m]
    ws: 0.05,               # Robotic arm short piece width [m]
    mlongarm: 0.510,        # Robotic Long arm piece mass [kg]
    mshortarm: 0.320,       # Robotic Short arm piece mass [kg]
    mbus: 121.2,            # Spacecraft bus mass [kg]
    }



# ----------------------------------- EVALUATING THE INERTIA MATRIX AT EACH TIME STEP -------------------------------- #
# ------------------------------------- CALCULATING THE PRINCIPAL INERTIA REF FRAME ---------------------------------- #
print('Calculating the principal inertia ref. frame...')

# Sostituzione dei valori costanti
Itot_baricentric_satellite = Itot_baricentric_satellite.xreplace(substitution)
Cn_b00 = Cn_b00.xreplace(substitution)

# Conversione della matrice di inerzia in una funzione numpy per velocizzare i calcoli
all_symbolic_var = (q1,q2,q3,q4,q7,q10,q13,q18,q21,q24,q27,q28,q31,q34,q37,q42,q45,q48,q51,q52,q53,q54)
Itot_baricentric_satellite_lambdified = sy.lambdify(all_symbolic_var, Itot_baricentric_satellite, 'numpy')
Cn_b00_lambdified = sy.lambdify(all_symbolic_var, Cn_b00, 'numpy')

# Lista per memorizzare le matrici di inerzia calcolate
Itot_baricentric_principal_sat_eval = []

# Soglia numerica per considerare i valori trascurabili come zero
threshold = 1e-4

for i in range(len(t_loaded)):
    # Valutazione della matrice di inerzia per il tempo i-esimo
    Itot_baricentric_satellite_lamb_eval = Itot_baricentric_satellite_lambdified(
        q1_data[i], q2_data[i], q3_data[i], q4_data[i], q7_data[i], q10_data[i], q13_data[i], q18_data[i], q21_data[i], q24_data[i], q27_data[i], q28_data[i], q31_data[i], q34_data[i], q37_data[i], q42_data[i], q45_data[i], q48_data[i], q51_data[i], q52_data[i], q53_data[i], q54_data[i]
    )

    Cn_b00_lamb_eval = Cn_b00_lambdified(
        q1_data[i], q2_data[i], q3_data[i], q4_data[i], q7_data[i], q10_data[i], q13_data[i], q18_data[i], q21_data[i], q24_data[i], q27_data[i], q28_data[i], q31_data[i], q34_data[i], q37_data[i], q42_data[i], q45_data[i], q48_data[i], q51_data[i], q52_data[i], q53_data[i], q54_data[i]
    )

    # Azzeramento dei valori piccoli per evitare problemi numerici
    Itot_baricentric_satellite_lamb_eval[np.abs(Itot_baricentric_satellite_lamb_eval) < threshold] = 0

    # Controllo se la matrice è già diagonale
    if np.allclose(Itot_baricentric_satellite_lamb_eval, np.diag(np.diagonal(Itot_baricentric_satellite_lamb_eval))):
        print(f"La matrice al tempo {t_loaded[i]} è già diagonale, si passa alla memorizzazione dei risultati...")

        # Memorizzazione dei risultati con matrice identità come autovettori
        Itot_baricentric_principal_sat_eval.append({
            't': t_loaded[i],
            'Inertia matrix': Itot_baricentric_satellite_lamb_eval.tolist(),
            'eigenvectors': np.eye(Itot_baricentric_satellite_lamb_eval.shape[0]).tolist(),
            'Cn_b00': Cn_b00_lamb_eval.tolist()
        })
        continue

    # Calcolo degli autovettori (SVD per ottenere base ortonormale)
    eigenvectors = np.linalg.svd(Itot_baricentric_satellite_lamb_eval)[0]

    # Rotazione della matrice di inerzia per ottenere la forma principale
    Itot_baricentric_principal_sat = eigenvectors.T @ Itot_baricentric_satellite_lamb_eval @ eigenvectors

    # Controllo dell'ortogonalità degli autovettori
    orthogonality_check = np.allclose(eigenvectors.T @ eigenvectors, np.eye(eigenvectors.shape[1]))
    print(f"Gli autovettori sono ortogonali? {orthogonality_check}")

    # Controllo se la matrice risultante è effettivamente diagonale
    is_diagonal = np.allclose(Itot_baricentric_principal_sat, np.diag(np.diagonal(Itot_baricentric_principal_sat)))
    print(f"La matrice trasformata è diagonale? {is_diagonal}")

    # Se la matrice non è ortogonale o la trasformazione non ha prodotto una matrice diagonale, interrompiamo il ciclo
    if not orthogonality_check or not is_diagonal:
        print("Errore: la trasformazione non è corretta, interruzione del ciclo.")
        break

    # Pulizia numerica dei valori trascurabili
    Itot_baricentric_principal_sat[np.abs(Itot_baricentric_principal_sat) < threshold] = 0

    # Memorizzazione dei risultati
    Itot_baricentric_principal_sat_eval.append({
        't': t_loaded[i],
        # 'Inertia matrix': Itot_baricentric_principal_sat.tolist(),
        'Inertia matrix': Itot_baricentric_satellite_lamb_eval.tolist(),
        'eigenvectors': eigenvectors.tolist(),
        'Cn_b00': Cn_b00_lamb_eval.tolist()
    })

    print(f"Step completato per t = {t_loaded[i]}")



# ------------------------------------- SAVING THE RESULTS INTO A JSON FILE ------------------------------- #
print('Saving results...')


# Saving the results
with open("Itot_baricentric_principal_sat_eval.json", "w") as f:
    json.dump(Itot_baricentric_principal_sat_eval, f, indent=4)



# ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ------------------------------ #

end = time.perf_counter()


# --------------------------------------------------------------------------------------------------------- #
# Showing the computation time
print()
print()
print(f"The calculations required time was: {end - start:.4f} seconds")
print('Codice terminato')















