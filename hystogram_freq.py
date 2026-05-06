
from datetime import datetime
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils.functions import Image2Array

VALORE_MIN = 1024

def collect_values(base_dir):
    values_reg = []
    values_TUE = []
    # Scorro cartelle train, test e validation
    for data_folder in os.listdir(base_dir):

        # Ignoro la cartella test2, dato che è un duplicato
        if data_folder == "test2":
            continue

        print(f"\nCartella {data_folder}")
        
        data_folder = os.path.join(base_dir, data_folder)
        # Scorro i vari AN dei tre dataset
        for an_folder in os.listdir(data_folder):

            print(f"\tCarico {an_folder}")
            an_path = os.path.join(data_folder, an_folder)

            # Cerco solo la cartella registration
            reg_path = os.path.join(an_path, "registration")

            if os.path.exists(reg_path):
                for file in os.listdir(reg_path):
                    # Appendo i valori dell'immagine "appiattita" su una dimensione
                    file_path = os.path.join(reg_path, file)
                    try:
                        img = np.load(file_path) + VALORE_MIN
                        values_reg.append(img.flatten())
                    except Exception as e:
                        print(f"Errore su {file_path}: {e}")

            # Cerco solo la cartella TUE
            tue_path = os.path.join(an_path, "TUE")

            if os.path.exists(tue_path):
                for file in os.listdir(tue_path):
                    # Appendo i valori dell'immagine "appiattita" su una dimensione
                    file_path = os.path.join(tue_path, file)
                    try:
                        img = Image2Array(file_path)
                        values_TUE.append(img.flatten())
                    except Exception as e:
                        print(f"Errore su {file_path}: {e}")

    return np.concatenate(values_reg), np.concatenate(values_TUE)

def collect_values_mask(reg_paths, reg_mask_paths, tue_paths, tue_mask_paths):
    values_reg = []
    values_TUE = []
    # Scorro cartelle train, test e validation
    for i in tqdm(range(len(reg_paths))):

        try:
            img = np.load(reg_paths[i]) + VALORE_MIN
            mask = np.load(reg_mask_paths[i])
            img = img * mask
            values_reg.append(img.flatten())
        except Exception as e:
            print(f"Errore su {reg_paths[i]} o {reg_mask_paths[i]}: {e}")


        # Appendo i valori dell'immagine "appiattita" su una dimensione
        try:
            img = Image2Array(tue_paths[i])
            mask = np.load(tue_mask_paths[i])
            img = img * mask
            values_TUE.append(img.flatten())
        except Exception as e:
            print(f"Errore su {tue_paths[i]} o {tue_mask_paths[i]}: {e}")

    return np.concatenate(values_reg), np.concatenate(values_TUE)


server = True

# percorso base
base_dir = Path(r"/mnt/raid/work/data/DECT_data")

if not server:
    base_dir = Path(r"/home/cei/Codice/test2")

data_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(data_dir, 'data')

datasetPaths_load = os.path.join(data_dir, "dataset_paths.pkl")
datasetPaths_val_load = os.path.join(data_dir, "dataset_paths_val.pkl")
datasetPaths_test_load = os.path.join(data_dir, "dataset_paths_test.pkl")

# Se i file non esistono ottengo errore specifico
if not os.path.exists(datasetPaths_load) or not os.path.exists(datasetPaths_val_load) or not os.path.exists(datasetPaths_test_load):
    raise FileNotFoundError(f"File dataset paths non trovati: \n{datasetPaths_load}\n{datasetPaths_val_load}\n{datasetPaths_test_load}\nEsegui prima con compute_dataset_paths = True")
print("\nCaricamento dataset paths\n")
# Carico i path dei pazienti di training
with open(datasetPaths_load, "rb") as f:
    DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath, AN_list_train = pickle.load(f)
# Carico i path dei pazienti di validation
with open(datasetPaths_val_load, "rb") as f:
    DSinputPath_val, DStargetPath_val, DSmaskPath_val, DSmaskRegPath_val, AN_list_val = pickle.load(f)

# Carico i path dei pazienti di test
with open(datasetPaths_test_load, "rb") as f:
    DSinputPath_test, DStargetPath_test, DSmaskPath_test, DSmaskRegPath_test, AN_list_test = pickle.load(f)

reg_paths = DSinputPath + DSinputPath_val + DSinputPath_test
reg_mask_paths = DSmaskRegPath + DSmaskRegPath_val + DSmaskRegPath_test
tue_paths = DStargetPath + DStargetPath_val + DStargetPath_test
tue_mask_paths = DSmaskPath + DSmaskPath_val + DSmaskPath_test
an_list = AN_list_train + AN_list_val + AN_list_test

actual_dir = os.path.dirname(os.path.realpath(__file__))
exp_path = os.path.join(actual_dir, 'experiments')

# Creazione cartella output grafico
img_dir = os.path.join(exp_path, 'istogrammi_frequenze_{}'.format(datetime.now().strftime("%d-%m-%Y--%H-%M")))

print("\nCaricamento dati")
reg_values, tue_values = collect_values_mask(reg_paths, reg_mask_paths, tue_paths, tue_mask_paths)

print(f"Numero pixel registration: {len(reg_values)}")
print(f"Numero pixel TUE: {len(tue_values)}")


# confronto istogrammi
plt.figure(figsize=(12, 6))

bins = 100

plt.hist(
    reg_values,
    bins=1000,
    range=(-100,2000),
    density=True,
    label="registration"
)

plt.hist(
    tue_values,
    bins=1000,
    range=(-100,2000),
    density=True,
    label="TUE"
)

plt.xlabel("Valore pixel")
plt.ylabel("Densità")
plt.title("Confronto distribuzione valori")
plt.legend()
plt.grid(True, alpha=0.3)

# Creo cartella dove salvare le immagini come ultima cosa
if not os.path.exists(img_dir):       
    os.mkdir(img_dir)

plt.savefig(os.path.join(img_dir, "istogramma_frequenze"), dpi=300, bbox_inches="tight")