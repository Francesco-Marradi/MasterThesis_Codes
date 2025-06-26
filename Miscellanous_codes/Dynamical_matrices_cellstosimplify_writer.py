
# Codice scritto per caricare le matrici dinamiche, e scrivere una lista al termine del file json
# delle celle (contenute nelle matrici) da semplificare.

# ----------------------------------------------------------------------------------------------- #

import sympy as sy
import json
import os

#Cleaning the terminal
os.system('cls')

print('Loading dynamical matrices...')


# Reading the Json file
with open('Dynamical_Matrices_Preloadedsprings_dampers_1Arm_OG.json', 'r') as json_file:
    data_Mat_Dyn = json.load(json_file)


# Reconstructing the Dynamical matrices
COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']['COEF_Dyn']))
RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']['RHS_Dyn']))



# Trova le celle non nulle
print('Writing the list of non-zero cells...')
print('')
print(f"The Mass matrix has dimensions of: {(COEF_Dyn_loaded.rows, COEF_Dyn_loaded.cols)}")

# Dal momento che la Mass matrix è simmetrica, tutte le celle [i,j] = [j,i] perciò usando il
# "for j in range(i,...) andiamo ad elencare tutte le celle sopradiagonali, diagonali comprese.
cells_coef = []
for i in range(COEF_Dyn_loaded.rows):
    for j in range(i, COEF_Dyn_loaded.cols): 
        if COEF_Dyn_loaded[i, j] != 0:
            cells_coef.append([i, j])

print('')
print(f"The Forcing vector matrix has dimensions of: {(RHS_Dyn_loaded.rows, RHS_Dyn_loaded.cols)}")
cells_rhs = []
for i in range(RHS_Dyn_loaded.rows):
    for j in range(RHS_Dyn_loaded.cols):
        if RHS_Dyn_loaded[i, j] != 0:
            cells_rhs.append([i, j])

input()
print('')
print('Saving datas...')
# Prepara il dizionario finale da salvare
output_data = {
    "COEF_Dyn": str(COEF_Dyn_loaded),
    "RHS_Dyn": str(RHS_Dyn_loaded),
    "Cells_to_Simplify": {
        "COEF": cells_coef,
        "RHS": cells_rhs
    }
}

# Salva in un nuovo file JSON
filename = "Dynamical_Matrices_Preloadedsprings_dampers_1Arm_with_Cells_to_simplify.json"

with open(filename, "w") as outfile:
    json.dump(output_data, outfile, indent=4)

print(f"In totale, vi saranno {len(cells_coef)+len(cells_rhs)} celle da semplificare")
print(f"File salvato correttamente in '{filename}'")