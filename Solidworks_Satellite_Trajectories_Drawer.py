#---------------------------------SCRIPT TEST PYTHON - FRANCESCO MARRADI-----------------------------------------------#

#------------------------------------ IMPORTING THE NECESSARY PACKAGES ----------------------------------------------- #

import win32com.client
import json
import sympy as sy
import numpy as np
import os


swApp = win32com.client.Dispatch("SldWorks.Application")
swApp.Visible = True

#Cleaning the terminal 
os.system('cls')

# --------------------------------------- LOADING THE DATAS ---------------------------------------------------------- #

# Body names list
bodies = ["B0", "B11", "B21", "B31", "B41", "G01", "G11", "G21", "G31", "B12", "B22", "B32", "B42",  "G02", "G12", "G22", "G32", "B13", "B23", "B33", "B43", "G03", "G13", "G23", "G33", "G04", "G14", "G24", "G34", "B14", "B24", "B34", "B44"] 

# Loading the file
with open("SolidworksAnimationDatas_1arm_fixedbus_torque_dg.json", "r") as f:
    data_raw = json.load(f)


# Extracting the values from the Json file

# Ricostruisci i dati originali convertendo le stringhe in oggetti Python
data = {}

for key, value in data_raw.items():

    # Extracting the time values
    if key == 't':
        # 1. Rimuovi le parentesi quadre e newline
        clean_string = data_raw['t'].replace('[', '').replace(']', '').replace('\n', ' ')

        # 2. Splitta la stringa in base agli spazi
        string_list = clean_string.split()

        # 3. Converte in float e crea l'array
        t_values = np.array([float(val) for val in string_list])

    else:
    
        # 1. Rimuovi le parentesi quadre e newline
        matrici_stringa = data_raw[key]

        # Usa sympy.sympify per convertire in lista di Matrix
        matrici_lista = sy.sympify(matrici_stringa)

        # Estrai ogni matrice e converti in lista di float
        data[key] = [list(map(float, m)) for m in matrici_lista]
        




# --------------------------------------- OPERATING ON SOLIDWORKS ---------------------------------------------------- #

# First thing first we need to get the names of each part of the assembly, and then associate it with the respective 
# values calculated with python

def trova_assieme_attivo(swApp):
    docs = swApp.GetDocuments  # Restituisce una tupla
    print(f"Documenti aperti: {len(docs)}")  # Stampiamo il numero di documenti aperti
    for doc in docs:  # Itera direttamente su docs
        print(f"Documento: {doc.GetTitle} - Tipo: {doc.GetType}")  # Stampiamo il nome e il tipo del documento
        if doc.GetType == 2:  # 2 = AssemblyDoc (corretto da 3 a 2)
            print(f"✅ Assieme trovato: {doc.GetTitle}")
            return doc
    print("❌ Nessun assieme aperto trovato.")
    return None

model = swApp.ActiveDoc

# Verifica se il documento attivo è un assieme
print(f"Documento attivo: {model.GetTitle} - Tipo: {model.GetType}")
if model.GetType != 2:  # Cambiato da 3 a 2 per il tipo di documento assieme
    print("⚠️ Il documento attivo non è un assieme. Cerco tra i documenti aperti...")
    model = trova_assieme_attivo(swApp)

if model is None or model.GetType!= 2:
    print("❌ Non riesco a trovare un assieme aperto.")
    exit()

assembly = model  # finalmente!

config = model.GetActiveConfiguration
root = config.GetRootComponent3(True)
all_components = root.GetChildren

components_by_name = {}

for comp in all_components:
    name = comp.Name2
    components_by_name[name] = comp
    print(f"📦 Componente trovato: {name}")



# === CREA SKETCH 3D CON SPLINE USANDO PUNTI TRASFORMATI NELL'ASSEMBLE ===
for body in bodies:
    print(f"✏️ Disegno della spline per: {body}...")

    # Entra in modalità modifica dell'assieme
    model.EditAssembly()

    # === CREA SKETCH 3D ===
    model.SketchManager.Insert3DSketch(True)

    punti_creati = []
    for coord in data[body]:
        x, y, z = coord

        # 🟢 Crea il punto in SolidWorks nell'assieme
        punto = model.SketchManager.CreatePoint(x, y, z)
        punti_creati.append(punto)

    # Chiude lo sketch 3D
    model.SketchManager.Insert3DSketch(False)

    # Esce dalla modifica dell'assieme
    model.EditAssembly()

    # 🔍 Ottieni l'ultima feature creata (che dovrebbe essere lo sketch appena chiuso)
    feature = model.FeatureManager.GetFeatures(True)[0]  # GetFeatures(True) restituisce le feature in ordine dal più recente
    feature.Name = f"Punti per {body}"  # Dai un nome allo sketch, ad esempio usando il nome del body

    input('crea la spline manualmente')
    # print(f"✅ Spline 3D per {body} creata con successo!\n")
    input(f'Associa la spline con: {body}')

print("🎉 Fatto! Le spline sono pronte nel sistema di riferimento inerziale!")