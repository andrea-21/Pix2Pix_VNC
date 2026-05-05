
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

        print(f"\nCartella {an_folder}")
        
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


server = True

# percorso base
base_dir = Path(r"/mnt/raid/work/data/DECT_data")

if not server:
    base_dir = Path(r"/home/cei/Codice/test2")


actual_dir = os.path.dirname(os.path.realpath(__file__))

exp_path = os.path.join(actual_dir, 'experiments')

# Creazione cartella output grafico
img_dir = os.path.join(exp_path, 'istogrammi_frequenze_{}'.format(datetime.now().strftime("%d-%m-%Y--%H-%M")))

print("\nCaricamento dati")
reg_values, tue_values = collect_values(base_dir)

print(f"Numero pixel registration: {len(reg_values)}")
print(f"Numero pixel TUE: {len(tue_values)}")


# confronto istogrammi
plt.figure(figsize=(12, 6))

bins = 100

plt.hist(
    reg_values,
    bins=bins,
    alpha=0.5,
    density=True,
    label="registration"
)

plt.hist(
    tue_values,
    bins=bins,
    alpha=0.5,
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