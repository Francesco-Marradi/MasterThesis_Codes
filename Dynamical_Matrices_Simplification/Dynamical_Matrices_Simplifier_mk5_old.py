#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI----------------------------------------------- #
#------------------------------------------- SCRIPT OBJECTIVE --------------------------------------------------------- #

# The objective of this script is to load the dynamical matrices, take the function in each cell and cut it in small
# more digestible pieces. Then operate a cycle of continuos simplification of this function. 

# -------------------------------------------------------------------------------------------------------------------- #

#------------------------------------ IMPORTING THE NECESSARY PACKAGES ----------------------------------------------- #

import sympy as sy
import os
import json
import time
from multiprocessing import Pool, Queue, Manager, Process
from multiprocessing import Process, Queue 
import multiprocessing as mp
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from concurrent.futures import TimeoutError, ThreadPoolExecutor
from datetime import datetime
import platform

# Setting the enviroment variable
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3"


# -------------------------------------------------------------------------------------------------------------------- #


def main():

    #Cleaning the terminal
    os.system('cls')

    # ------------------------------------------------ SEMPLIFICATION SETTINGS ----------------------------------------- #
    max_cpus = mp.cpu_count()
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

    # Lunghezza delle sotto-espressioni da semplificare [usare "None" per chunk_size adattiva]
    chunk_size = None #300 

    # Cicli di semplificazione
    max_iter = 5

    # Soglia per definire il rateo di semplificazione fra un ciclo e il successivo
    threshold = 0.1 # 10%

    # Nome del file da caricare
    filename = "Dynamical_Matrices_Preloadedsprings_dampers_1Arm_BusFixed_with_Cells_to_simplify.json"

    # Attivare lo spegnimento del pc con salvataggio del log dei tempi al termine
    saving_shuttingdown = True



    # ---------------------------------- STARTING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------- #

    start = time.perf_counter()

    # ---------------------------------------- LOADING THE DYNAMICAL MATRICES -------------------------------------------- #
    print('Loading dynamical matrices...')


    # Reading the Json file
    with open(filename, 'r') as json_file:
        data_Mat_Dyn = json.load(json_file)


    # Reconstructing the Dynamical matrices
    COEF_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['COEF_Dyn']))
    RHS_Dyn_loaded = sy.Matrix(sy.sympify(data_Mat_Dyn['RHS_Dyn']))

    cells_to_simplify_COEF = data_Mat_Dyn.get('Cells_to_Simplify', {}).get('COEF', [])
    cells_to_simplify_RHS = data_Mat_Dyn.get('Cells_to_Simplify', {}).get('RHS', [])



    # ------------------------------------ PHYSICAL PROBLEM DEFINITION ----------------------------------------------- # 
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


    print('Simplifying the Mass Matrix...')
    print('')
    print('')
    COEF_Symplified, flagged_cells_COEF = parallel_simplify_matrix(COEF_Dyn_loaded, cells_to_simplify_COEF, num_processes, threshold, max_iter, chunk_size)

    print('')
    print('')

    print('Simplifying the Forcing vector...')
    print('')
    print('')
    RHS_Symplified, flagged_cells_RHS = parallel_simplify_matrix(RHS_Dyn_loaded, cells_to_simplify_RHS, num_processes, threshold, max_iter, chunk_size)



#  -------------------------------------------------- PYTHON NECESSITIES ------------------------------------------- #

    # Salvataggio
    print('\n💾 Salvataggio file aggiornato...')
    output_data = {
            'COEF_Dyn_simplified': str(COEF_Symplified),
            'RHS_Dyn_simplified': str(RHS_Symplified),
                'Cells_to_Simplify': {
                    'COEF': flagged_cells_COEF,
                    'RHS': flagged_cells_RHS
                    }
                }


    # Writing and saving these dictionaries in a Json file
    with open('Dynamical_Matrices_Preloadedsprings_dampers_1Arm_simplified.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=2)



    # ---------------------------- STOPPING THE CLOCK FOR PERFORMANCE EVALUATION ----------------------------------------- #
    end = time.perf_counter()

    # Showing the computation time
    print(f"The calculations required time was: {end - start:.4f} seconds")
    print()
    print('Codice terminato')


    # Attivare o meno lo shutting down del computer una volta terminati i calcoli
    if saving_shuttingdown == False:
        
        # ⏱️ Salvataggio tempo in un file
        with open("tempo_semplificazione.txt", "w") as f:
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            f.write(f"Fine esecuzione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tempo totale: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")

        # 🔻 Spegnimento forzato (solo se su Windows)
        if platform.system() == "Windows":
            print("⚠️ Spegnimento del sistema in corso...")
            os.system("shutdown /s /f /t 60")  # /s = shutdown, /f = forza chiusura, /t 60 = 60 secondi#




# --------------------------------------------- SEMPLIFICATION FUNCTIONS --------------------------------------------- #

def wrapped_process_cell_safe(args_queue_tuple):
    args, queue = args_queue_tuple
    i, j, *_ = args
    try:
        i, j, simplified = wrapped_process_cell(args_queue_tuple)
        return i, j, simplified
    except TimeoutError:
        if queue:
            queue.put(((i, j), "FLAGGED")) 
        return i, j, "FLAGGED"
    

def split_expression(expr, chunk_size, iteration):
    """
    Spezza una espressione somma lunga in sotto-espressioni ogni chunk_size caratteri circa.
    Si divide solo in corrispondenza di "+".

    In pratica questa funzione procede a spezzare l'espressione espansa ogni volta che trova un +. 
    Una volta fatto ciò, aggiunge un pezzettino per volta e controlla che la sotto-espressione che sta
    costruendo non superi la lunghezza scelta (chunk_size). Appena questo accade, la sotto-espressione
    è completata e procede a formarne una nuova.

    - Se il termine "average_term_length" < 500 -> il coefficiente moltiplicativo che sta davanti ad "average_term_length" è 1.25. 
    - Se invece il termine "average_term_length" è maggiore di 500 ma minore di 1000 allora il coefficiente moltiplicativo che 
      sta davanti ad "average_term_length" è 0.85.
    - Se invece il termine "average_term_length" è maggiore 1000 allora vorrei che il coefficiente moltiplicativo che sta davanti ad 
      "average_term_length" è 0.5.

    In tutti e tre i casi, nel caso in cui il singolo termine lavorato (term) sia più lungo di 1000 caratteri (len(str(term)) >1000, allora
    "espanso", usando .expand(), creando cosi nuovi "terms" da accorpare successivamente, creando sotto-espressioni più piccole e facili da gestire.

    """
    terms = list(sy.Add.make_args(expr))

    # Calcolo automatico chunk_size se non fornito
    if chunk_size is None:
        term_lengths = [len(str(term)) for term in terms]
        average_term_length = sum(term_lengths) / len(term_lengths)

        # Pezzettino di codice che modifica il chunksize a seconda dell ciclo semplificativo
        if iteration == 1:
            coeff = 0.1
        else:
            coeff = 1.6

    chunk_size = int(coeff * average_term_length)

    # Suddivisione in chunk
    chunks = []
    current_chunk = []

    for term in terms:
        current_chunk.append(term)
        chunk_str = str(sy.Add(*current_chunk))
        if len(chunk_str) >= chunk_size:
            chunks.append(sy.Add(*current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(sy.Add(*current_chunk))

    return chunks



def simplify_expression(expr, threshold, max_iter, chunk_size, i=None, j=None, queue=None, simplification_cache=None):
    """
    Semplifica ricorsivamente una espressione simbolica spezzandola in chunk.
    La semplificazione si ferma quando la riduzione di lunghezza è sotto threshold.

    Per prima cosa la funzione espande l'espressione simbolica complessa, e la misura. Poi comincia
    il vero ciclo semplficativo. Infatti, la funzione usa la funzione "split_expression" per ricavare
    il pezzettino di sotto-espressione con cui lavorare. Questo ultimo è una stringa (dalla manipolazione
    di "split_expression"), perciò per prima cosa viene convertito in quantità simboliche con "sympify" e
    successivamente semplificato con "symplify". 
    Dopo che tutte le sotto-espressioni sono state semplificate, viene ricomposta l'espressione grande e la
    sua lunghezza viene valutata per confrontare il rateo di "semplificazione".
    Se tale rateo scende sotto un valore di soglia (threshold a 0,1 ->10%) allora l'espressione è già abbastanza
    semplificata, perciò viene salvata. Altrimenti il ciclo ricomincia fino ad un max di (max_iter) cicli si 
    semplificazione. 

    Questa versione della funzione introduce il caching locale dei terms semplificati e un timeout per la semplificazione 
    dei chunks, in modo da non intoppare troppo a lungo il processo di semplificazione.
    """
    if simplification_cache is None:
            simplification_cache = {}

    original_expr = sy.sympify(expr)
    prev_len = len(str(original_expr))

    for iteration in range(1, max_iter + 1):
        try:
            if queue:
                queue.put(((i, j), "Splitting"))
                
            chunks = split_expression(original_expr, chunk_size, iteration)
        except Exception as e:
            print(f"❌ Errore nella suddivisione dell'espressione [{i},{j}]: {e}")
            raise

        simplified_chunks = []

        for idx, chunk in enumerate(chunks, start=1):
            chunk_key = str(chunk)

            if queue:
                queue.put(((i, j), f"chunk {idx}/{len(chunks)}"))

            if chunk_key in simplification_cache:
                simplified_chunk = simplification_cache[chunk_key]
            else:
                simplified_chunk = simplify_with_timeout(iteration, chunk, i, j, queue, timeout=10)

                if simplified_chunk == "FLAGGED":
                    return original_expr
                
                simplification_cache[chunk_key] = simplified_chunk

            simplified_chunks.append(simplified_chunk)

        recomposed = sy.Add(*simplified_chunks)
        new_len = len(str(recomposed))
        rate = (prev_len - new_len) / prev_len if prev_len != 0 else 0

        if rate < threshold:
            if queue:
                queue.put(((i, j), "DONE"))
                queue.put("PROGRESS")
            return recomposed

        original_expr = recomposed
        prev_len = new_len

    if queue:
        queue.put(((i, j), "DONE"))
        queue.put("PROGRESS")

    return original_expr



def process_cell(args):
    """
    E' una funzione molto semplice che ha il compito di caricare la posizione [i,j] dell'espressione
    da semplificare e la sua espressione (expr). Questa ultima viene poi passata alla funzione
    "simplify_expression", in modo che la semplifichi. La funzione semplificata, poi viene restituita
    insieme alla posizione [i,j] della funzione originaria.
    """
    i, j, expr, threshold, max_iter, chunk_size = args
    simplified = simplify_expression(expr, threshold, max_iter, chunk_size, i=i, j=j)
    return i, j, simplified


def wrapped_process_cell(args_queue_tuple):
    args, queue = args_queue_tuple
    i, j, expr, threshold, max_iter, chunk_size = args
    simplified = simplify_expression(expr, threshold, max_iter, chunk_size, i=i, j=j, queue=queue)
    return i, j, simplified


def parallel_simplify_matrix(matrix, cells_to_simplify, num_processes, threshold, max_iter, chunk_size):
    """s
    E' la funzione madre, colei che ha il compito di smistare il "carico di lavoro" parallellizzando il tutto
    sui diversi core disponibili.
    Per prima cosa, crea una lista di task da fare (jobs), le quali altro non sono che tuple contenenti la 
    posizione [i,j] della espressione da semplficare e la sua espressione vera e propria (come sopra "i,j,expr")
    Fatto ciò, queste task vengono smistate sui core disponibili, i quali si occupano di semplificare ognuno una
    cella per volta.
    Viene poi creata una nuova matrice sympy avente le dimensioni di quella originale, ma contenente solo zeri.
    Tale matrice viene poi riempita con le espressioni semplificate, man mano che queste vengono ricavate.
    """

    jobs = [(i, j, matrix[i, j], threshold, max_iter, chunk_size) 
        for i, j in cells_to_simplify]

    results = []
    flagged_cells = []
    manager = Manager()
    queue = manager.Queue()
    progress_proc = Process(target=monitor_progress, args=(queue, len(jobs)))
    progress_proc.start()

    job_args_with_queue = [(args, queue) for args in jobs]

    with Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(wrapped_process_cell_safe, job_args_with_queue):
            if result is None:
                continue
            i, j, simplified_or_flag = result
            if simplified_or_flag == "FLAGGED":
                flagged_cells.append([i, j])
                # Salva comunque il valore originale
                results.append((i, j, matrix[i, j]))  
            else:
                results.append((i, j, simplified_or_flag))

    queue.put("DONE")
    progress_proc.join()


    # Poichè la mass matrix è simmetrica, andiamo a sostituire l'espressione semplificata in entrambe 
    # le celle speculari, sempre se esistano (cosa che non accade nel caso stiamo semplificando il 
    # forcing vector)
    Simplified_matrix = sy.MutableDenseMatrix(matrix.rows, matrix.cols, [0]*matrix.rows*matrix.cols)
    for i, j, simplified in results:
        Simplified_matrix[i, j] = simplified
        if i != j and j < matrix.rows and i < matrix.cols:
            Simplified_matrix[j, i] = simplified

    return Simplified_matrix, flagged_cells
    



def simplify_worker(iteration, chunk, output_queue):
    try:
        if iteration == 1:
            result = sy.simplify(chunk)
        else:
            result = sy.trigsimp(chunk)
        output_queue.put(result)
    except Exception as e:
        output_queue.put(e)


def simplify_with_timeout(iteration, chunk, i, j, queue, timeout):
    def worker():
        if iteration == 1:
            return sy.simplify(chunk)
        else:
            return sy.trigsimp(chunk)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            # Timeout: restituisci chunk originale o flagga
            if queue:
                queue.put(((i,j), "FLAGGED"))
            return "FLAGGED"




# ---------------------------------------------- DISPLAYING FUNCTIONS ---------------------------------------------- #


def monitor_progress(queue: Queue, total_cells: int):
    """Barra e tabella insieme usando Rich, con visualizzazione celle flaggate che spariscono dopo timeout."""

    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table

    import time

    FLAGGED_DISPLAY_SECONDS = 60  # ⏲️ Tempo di visualizzazione per le celle FLAGGED

    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    task = progress_bar.add_task("Simplification progress", total=total_cells)

    progress_dict = {}
    flagged_timestamps = {}  # 🆕 Tieni traccia di quando ogni cella è stata flaggata
    flagged_count = 0
    completed_cells = set()

    def render_layout():
        table = Table(title="Cells status")
        table.add_column("Cella", justify="center")
        table.add_column("Stato", justify="left")

        # Mostra solo celle attualmente attive o recentemente flaggate
        for key, status in sorted(progress_dict.items()):
            table.add_row(f"[{key[0]},{key[1]}]", status)

        flagged_info = f"[red]Flagged cells: {flagged_count} / {total_cells}[/red]"

        return Panel.fit(
            Group(progress_bar, table, flagged_info),
            title="📊 Simplification in progress...",
            border_style="cyan"
        )

    with Live(render_layout(), refresh_per_second=5) as live:
        while True:
            # 🕒 Rimuovi celle flaggate da troppo tempo
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in flagged_timestamps.items()
                if current_time - timestamp > FLAGGED_DISPLAY_SECONDS
            ]
            for key in expired_keys:
                progress_dict.pop(key, None)
                flagged_timestamps.pop(key, None)
                live.update(render_layout())

            if len(completed_cells) >= total_cells:
                break

            try:
                msg = queue.get(timeout=1)
            except:
                continue

            if isinstance(msg, str) and msg == "DONE":
                break

            if isinstance(msg, tuple) and len(msg) == 2:
                key, status = msg

                if status == "DONE":
                    progress_dict.pop(key, None)
                    flagged_timestamps.pop(key, None)
                    completed_cells.add(key)
                    progress_bar.update(task, advance=1)

                elif status == "FLAGGED":
                    flagged_count += 1
                    progress_dict[key] = "[red]FLAGGED[/red]"
                    flagged_timestamps[key] = time.time()
                    completed_cells.add(key)
                    progress_bar.update(task, advance=1)

                else:
                    progress_dict[key] = status

                live.update(render_layout())





# Impedisce allo script di essere richiamato in perpetuo
if __name__ == "__main__":
    main()