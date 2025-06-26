#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI----------------------------------------------- #
#------------------------------------------- SCRIPT OBJECTIVE --------------------------------------------------------- #

# The objective of this script is to load the dynamical matrices, take the function in each cell and cut it in small
# more digestible pieces. Then operate a cycle of continuos simplification of this function. 

# -------------------------------------------------------------------------------------------------------------------- #

#------------------------------------ IMPORTING THE NECESSARY PACKAGES ----------------------------------------------- #

import sympy as sy
import re
import os
import json
import time
from multiprocessing import Pool,TimeoutError
import multiprocessing
from tqdm import tqdm


# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"

# Cache globale per chunks già semplificati
simplify_cache = {}

# -------------------------------------------------------------------------------------------------------------------- #


def main():

    #Cleaning the terminal
    os.system('cls')

    # ------------------------------------------------ SEMPLIFICATION SETTINGS ----------------------------------------- #
    max_cpus = multiprocessing.cpu_count()
    print(f"Numero di CPU logiche disponibili: {max_cpus}")

    while True:
        try:
            user_input = input(f"Inserisci il numero di CPU da usare [1 - {max_cpus}]: ")
            num_processes = int(user_input)
            if 1 <= num_processes <= max_cpus:
                break
            else:
                print(f"❌ Inserisci un numero tra 1 e {max_cpus}.")
        except ValueError:
            print("❌ Inserisci un numero por favor!")

    print(f"Numero di CPU logiche dedicate al lavoro: {num_processes}\n")

    # Lunghezza caratteri max delle sotto-espressioni da semplificare
    chunk_size = None

    # Cicli di semplificazione
    max_iter = 5

    # Soglia per definire il rateo di semplificazione fra un ciclo e il successivo
    threshold = 0.1 # 10%



    # ---------------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------- #

    start = time.perf_counter()

    # ---------------------------------------- LOADING THE DYNAMICAL MATRICES -------------------------------------------- #
    print('Loading dynamical matrices...')
    with open('Dynamical_Matrices_Preloadedsprings_dampers_1Arm_with_Cells_to_simplify44.json', 'r') as json_file:
        data_Mat_Dyn = json.load(json_file)

    # Caricamento matrici
    COEF_Dyn = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn_cell44']))
    # RHS_Dyn = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']))

    # Celle da semplificare
    cells_to_simplify_COEF = data_Mat_Dyn.get('Cells_to_Simplify', {}).get('COEF', [])
    # cells_to_simplify_RHS = data_Mat_Dyn.get('Cells_to_Simplify', {}).get('RHS', [])


    #------------------------------------------ SYMBOLIC VARIABLES ----------------------------------------------------#    
    print('Defining the symbols contained into the matrices...')


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


    # Forces and Torques applied
    T11 = sy.symbols('T11')             # Torque acting on the arms [N*m]
    T21 = sy.symbols('T21')             # Torque acting on the arms [N*m]
    T31 = sy.symbols('T31')             # Torque acting on the arms [N*m]
    T41 = sy.symbols('T41')             # Torque acting on the arms [N*m]
    T12 = sy.symbols('T12')             # Torque acting on the arms [N*m]
    T22 = sy.symbols('T22')             # Torque acting on the arms [N*m]
    T32 = sy.symbols('T32')             # Torque acting on the arms [N*m]
    T42 = sy.symbols('T42')             # Torque acting on the arms [N*m]
    T13 = sy.symbols('T13')             # Torque acting on the arms [N*m]
    T23 = sy.symbols('T23')             # Torque acting on the arms [N*m]
    T33 = sy.symbols('T33')             # Torque acting on the arms [N*m]
    T43 = sy.symbols('T43')             # Torque acting on the arms [N*m]
    T14 = sy.symbols('T14')             # Torque acting on the arms [N*m]
    T24 = sy.symbols('T24')             # Torque acting on the arms [N*m]
    T34 = sy.symbols('T34')             # Torque acting on the arms [N*m]
    T44 = sy.symbols('T44')             # Torque acting on the arms [N*m]
    F = sy.symbols('F')                 # Force acting on the arms [N*m]


    # Forces and Torques constants
    k11 = sy.symbols('k11')             # B11 body spring's stiffness [N*m/rad]
    k21 = sy.symbols('k21')             # B21 body spring's stiffness [N*m/rad] 
    k31 = sy.symbols('k31')             # B31 body spring's stiffness [N*m/rad] 
    k41 = sy.symbols('k41')             # B41 body spring's stiffness [N*m/rad]
    k12 = sy.symbols('k12')             # B12 body spring's stiffness [N*m/rad]
    k22 = sy.symbols('k22')             # B22 body spring's stiffness [N*m/rad] 
    k32 = sy.symbols('k32')             # B32 body spring's stiffness [N*m/rad] 
    k42 = sy.symbols('k42')             # B42 body spring's stiffness [N*m/rad] 
    k13 = sy.symbols('k13')             # B13 body spring's stiffness [N*m/rad]
    k23 = sy.symbols('k23')             # B23 body spring's stiffness [N*m/rad] 
    k33 = sy.symbols('k33')             # B33 body spring's stiffness [N*m/rad] 
    k43 = sy.symbols('k43')             # B43 body spring's stiffness [N*m/rad] 
    k14 = sy.symbols('k14')             # B14 body spring's stiffness [N*m/rad]
    k24 = sy.symbols('k24')             # B24 body spring's stiffness [N*m/rad] 
    k34 = sy.symbols('k34')             # B34 body spring's stiffness [N*m/rad] 
    k44 = sy.symbols('k44')             # B44 body spring's stiffness [N*m/rad] 

    c11 = sy.symbols('c11')             # B11 body damping coefficient [(N*m/s)/(rad/s)]
    c21 = sy.symbols('c21')             # B21 body damping coefficient [(N*m/s)/(rad/s)]
    c31 = sy.symbols('c31')             # B31 body damping coefficient [(N*m/s)/(rad/s)]
    c41 = sy.symbols('c41')             # B41 body damping coefficient [(N*m/s)/(rad/s)]
    c12 = sy.symbols('c12')             # B12 body damping coefficient [(N*m/s)/(rad/s)]
    c22 = sy.symbols('c22')             # B22 body damping coefficient [(N*m/s)/(rad/s)]
    c32 = sy.symbols('c32')             # B32 body damping coefficient [(N*m/s)/(rad/s)]
    c42 = sy.symbols('c42')             # B42 body damping coefficient [(N*m/s)/(rad/s)]
    c13 = sy.symbols('c13')             # B13 body damping coefficient [(N*m/s)/(rad/s)]
    c23 = sy.symbols('c23')             # B23 body damping coefficient [(N*m/s)/(rad/s)]
    c33 = sy.symbols('c33')             # B33 body damping coefficient [(N*m/s)/(rad/s)]
    c43 = sy.symbols('c43')             # B43 body damping coefficient [(N*m/s)/(rad/s)]
    c14 = sy.symbols('c14')             # B14 body damping coefficient [(N*m/s)/(rad/s)]
    c24 = sy.symbols('c24')             # B24 body damping coefficient [(N*m/s)/(rad/s)]
    c34 = sy.symbols('c34')             # B34 body damping coefficient [(N*m/s)/(rad/s)]
    c44 = sy.symbols('c44')             # B44 body damping coefficient [(N*m/s)/(rad/s)]


    #  ---------------------------------------------- #
    # Initializing the gen.coordinates and speed
    q1, q2, q3, q4, q7, q10, q13, q18, q21, q24, q27, q28, q31, q34, q37, q42, q45, q48, q51, q52, q53, q54 = sy.symbols('q1 q2 q3 q4 q7 q10 q13 q18 q21 q24 q27 q28 q31 q34 q37 q42 q45 q48 q51 q52 q53 q54')
    u1, u2, u3, u4, u7, u10, u13, u18, u21, u24, u27, u28, u31, u34, u37, u42, u45, u48, u51, u52, u53, u54 = sy.symbols('u1 u2 u3 u4 u7 u10 u13 u18 u21 u24 u27 u28 u31 u34 u37 u42 u45 u48 u51 u52 u53 u54')



    # --------------------------------------- LAVORIAMO SOLO SU UNA CELLA ------------------------------------------- #
    if not cells_to_simplify_COEF and not cells_to_simplify_RHS:
        print("✅ Nessuna cella da semplificare. Tutto già fatto!")
        return

    print("🎯 Lavoro mirato su UNA sola cella per volta.")

    updated_cells_COEF = []
    updated_cells_RHS = []


    if cells_to_simplify_COEF:
        cell = cells_to_simplify_COEF[0]
        print(f"\n🔧 Semplificazione Matrice COEF. Rimangono {len(cells_to_simplify_COEF)} celle da semplificare")
        print(f"🔧 Inizio semplificazione singola cella COEF {cell}...")
        COEF_Dyn, updated_cells_COEF = parallel_simplify_specific_cells(
            COEF_Dyn, [cell], num_processes, threshold, max_iter, chunk_size
        )
    # elif cells_to_simplify_RHS:
    #     cell = cells_to_simplify_RHS[0]
    #     print(f"\n🔧 Semplificazione Matrice RHS. Rimangono {len(cells_to_simplify_RHS)} celle da semplificare")
    #     print(f"🔧 Inizio semplificazione singola cella RHS {cell}...")
    #     RHS_Dyn, updated_cells_RHS = parallel_simplify_specific_cells(
    #         RHS_Dyn, [cell], num_processes, threshold, max_iter, chunk_size
    #     )


    # Rimozione celle completate
    remaining_cells_COEF = [cell for cell in cells_to_simplify_COEF if cell not in updated_cells_COEF]
    # remaining_cells_RHS = [cell for cell in cells_to_simplify_RHS if cell not in updated_cells_RHS]




#  -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #
    print('Saving datas...')

    # Salvataggio
    print('\n💾 Salvataggio file aggiornato...')
    output_data = {
            'COEF_Dyn': str(COEF_Dyn),
            # 'RHS_Dyn': str(RHS_Dyn),
                'Cells_to_Simplify': {
                    'COEF': remaining_cells_COEF,
                    # 'RHS': remaining_cells_RHS
                    }
                }

    with open('Clearspace_Dynamical_Matrices_with_Cells_Progressive.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=2)


    # ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #
    end = time.perf_counter()

    # Showing the computation time
    print(f"\n⏱️ Tempo totale: {end - start:.2f} secondi")
    print("✅ Script completato!")




# --------------------------------------------- SEMPLIFICATION FUNCTIONS --------------------------------------------- #

def simplify_chunk(chunk):
    """Semplifica un chunk senza cache, per chiamate in pool."""
    return sy.simplify(chunk)

def simplify_chunk_cached(chunk):
    """
    Semplifica un chunk, usando cache per evitare di ripetere calcoli.
    """
    key = str(chunk)
    if key in simplify_cache:
        return simplify_cache[key]

    result = simplify_chunk(chunk)
    simplify_cache[key] = result
    return result


def split_expression(expr, chunk_size, iteration):
    """
    Spezza una espressione somma lunga in sotto-espressioni ogni `chunk_size` termini circa.
    """
    terms = sy.Add.make_args(expr)
    chunks = []
    current_chunk = []

    if iteration == 1:
        chunk_size = 50
    else:
        chunk_size = 300

    for term in terms:
        current_chunk.append(term)
        if len(str(current_chunk)) >= chunk_size:
            chunks.append(sy.Add(*current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(sy.Add(*current_chunk))

    return chunks


def simplify_expression(expr, threshold, max_iter, chunk_size, i=None, j=None, parallel=True, num_processes=None, timeout=300):
    """
    Semplifica ricorsivamente un'espressione, con cache e pool di processi con timeout per ogni chunk.
    """
    original_expr = sy.sympify(expr)
    prev_len = len(str(original_expr))

    for iteration in range(1, max_iter + 1):
        simplify_cache.clear()

        print(f" Numero cicli di semplificazione: {iteration} su {max_iter}")
        try:
            chunks = split_expression(original_expr, chunk_size, iteration)
        except Exception as e:
            print(f"❌ Errore nella suddivisione dell'espressione [{i},{j}]: {e}")
            raise

        simplified_chunks = []

        if parallel:
            with Pool(processes=min(num_processes or os.cpu_count(), len(chunks))) as pool:
                
                results = []
                
                for chunk in chunks:
                    res = pool.apply_async(simplify_chunk, (chunk,))
                    results.append((res, chunk))
                
                for res, chunk in tqdm(results, desc="🔧 Semplificazione chunk"):
                    try:
                        simplified = res.get(timeout=timeout)
                        simplified_chunks.append(simplified)
                    except TimeoutError:
                        print("Chunk troppo difficile, lo lascio invariato.")
                        simplified_chunks.append(chunk)

        else:
            for chunk in tqdm(chunks, desc="🔧 Semplificazione chunk"):
                simplified = simplify_chunk_cached(chunk)
                simplified_chunks.append(simplified)

        recomosed = sy.Add(*simplified_chunks)
        new_len = len(str(recomosed))
        rate = (prev_len - new_len) / prev_len if prev_len != 0 else 0

        print(f"🔁 Iter {iteration} | cella [{i},{j}] | rate: {rate:.3f} | soglia: {threshold}")

        if rate < threshold:
            return recomosed

        original_expr = recomosed
        prev_len = new_len

    return original_expr


def process_cell(args):
    """
    Processa una singola cella: semplifica l'espressione con le opzioni date.
    """
    i, j, expr, threshold, max_iter, chunk_size, parallel, num_processes = args
    simplified = simplify_expression(expr, threshold, max_iter, chunk_size, i=i, j=j,
                                     parallel=parallel, num_processes=num_processes)
    return i, j, simplified


def parallel_simplify_specific_cells(matrix, cells_to_simplify, num_processes, threshold, max_iter, chunk_size):
    """
    Semplifica solo le celle indicate in cells_to_simplify, con parallelizzazione a livello di cella.
    """
    jobs = [
        (i, j, matrix[i, j], threshold, max_iter, chunk_size, True, num_processes)
        for i, j in cells_to_simplify
    ]

    results = []

    for job in jobs:
        result = process_cell(job)
        results.append(result)

    updated_matrix = sy.MutableDenseMatrix(matrix.rows, matrix.cols, list(matrix))
    completed_cells = []

    for i, j, simplified in results:
        updated_matrix[i, j] = simplified
        if i != j and j < matrix.rows and i < matrix.cols:
            if j < matrix.rows and i < matrix.cols:
                 updated_matrix[j, i] = simplified

        completed_cells.append([i, j])

    return updated_matrix, completed_cells



# Impedisce allo script di essere richiamato in perpetuo
if __name__ == "__main__":
    main()