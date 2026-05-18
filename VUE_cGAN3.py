# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 17:07:17 2025

@author: Marco
"""

#%% Modules and GPU

import sys

import tensorflow as tf
#from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import skimage
#from skimage import color
#from skimage import io

#from tensorflow.keras.models import Model

from pydicom import dcmread
from pathlib import Path
import os
import math
#from sklearn.model_selection import train_test_split
import json
import gc
from datetime import datetime
import shutil
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import pandas as pd
import time
from IPython import display
import secrets
from scipy.ndimage import uniform_filter
import pickle

start_program = time.time()

server = True

if server:

    gpus = tf.config.list_physical_devices('GPU')
   
    import subprocess as sp


    def get_gpu_memory():
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values
   
    GPUs_memory = get_gpu_memory()
   
    # USARE 32 per batch da 32
    # USARE 22 per batch da 8
    memory_limit = 32*1024
    # lista dell'indice della GPU con memoria libera maggiore del valore richiesto
    GPU_index = [i for i, v in enumerate(GPUs_memory) if v > memory_limit]
    # prendo il valore scalare da dentro la lista
    GPU_index = GPU_index[0]
   
       
    if gpus:
      # Restrict TensorFlow to only allocate 25GB of memory on the first GPU
      try:
        tf.config.set_visible_devices(gpus[GPU_index], 'GPU') #singola GPU
        tf.config.set_logical_device_configuration(
            gpus[GPU_index],
            [tf.config.LogicalDeviceConfiguration(memory_limit = memory_limit)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)   


#%% Parameters
from utils.functions import *
from utils.dataset import *
from models.generator import *
from models.discriminator import *
from utils.train import *

#network parameters


batch_size = 8
# con 32 sono 409 (10 circa a epoca)
# con 4 sono 3273 (circa) (15 circa a epoca)
# con 8 sono 1637 (0.7s a batch/10.3s per 15 batch) (19 minuti per epoca)
# secondo lancio con 8:  (0.33s a batch/4.89s per 15 batch) (9 minuti per epoca)

# Scelta se riottenre i path delle immagini valide
compute_dataset_paths = False 

# Scelta se ricalcorare i parametri dei pazienti
compute_patient_parameters = False 

# Scelta numero canali (1 "standard", 3 o piu' "2.5D")
n_canali = 1

# Scelta valore di stride per il dataset 2.5D
stride = 1

#########################
####### TRAINING ########
#########################
train = False
#######################


#########################
#### NORMALIZATION ######
#########################
normalizeByPatient = True
normalizationFunction = ['standard']
#########################


#########################
######## OTHER ##########
#########################
plot = False
statistics = False
#########################

epochs = 400
_lambda = 100

#########################
####### AUTO BOOL #######
#########################
test = not train
#########################


#file paths

file_path = os.path.realpath(__file__)
checkpoint_dir = os.path.dirname(os.path.realpath(__file__))

checkpoint_dir = os.path.join(checkpoint_dir, 'experiments')

# CREO CARTELLA EXPERIMENTO
if train:
    experimentPath = os.path.join(checkpoint_dir, 'experiment_{}'.format(datetime.now().strftime("%d-%m-%Y--%H-%M")))
else:
    # experimentPath per indicare dove salvare parametri i path (se li sto calcolando) e metto plotpath dentro questa cartella
    experimentPath = os.path.join(checkpoint_dir, 'experiment_test_{}'.format(datetime.now().strftime("%d-%m-%Y--%H-%M")))
    plotPath = os.path.join(experimentPath, "plots")

if not os.path.exists(experimentPath):       
    os.mkdir(experimentPath)


# Salvo "immagine" del progetto usata per fare l'esperimento
source_root = os.path.dirname(os.path.realpath(__file__))
destination_root = os.path.join(experimentPath, "project")

for dirpath, dirnames, filenames in os.walk(source_root):

    # Salta cartella experiments
    if "experiments" in dirnames:
        dirnames.remove("experiments")

    # calcola il path relativo da "project"
    relative_path = os.path.relpath(dirpath, source_root)

    # costruisce la directory di destinazione
    target_dir = os.path.join(destination_root, relative_path)
    os.makedirs(target_dir, exist_ok=True)

    for file in filenames:

        source_path = os.path.join(dirpath, file)

        is_py = file.endswith(".py")
        is_data_pkl = (
            "data" in dirpath.split(os.sep) and file.endswith(".pkl")
        )

        if is_py or is_data_pkl:
            destination_path = os.path.join(target_dir, file)
            shutil.copy2(source_path, destination_path)

print("\nCopiato tutto il modello in ", experimentPath, "\n")
        
# Train path
'''patientFolder = Path(r"/home/rossetti/DECT_data/train")
if server == False:
    patientFolder = Path(r"D:\Tesi_magistrale\Codice\test2")'''

patientFolder = Path(r"/mnt/raid/work/data/DECT_data/train")
if not server:
    patientFolder = Path(r"/home/cei/Codice/test2")

# Validation path
'''patientFolder_val = Path(r"/home/rossetti/DECT_data/validation")
if server == False:
    patientFolder_val = Path(r"D:\Tesi_magistrale\Codice\test2")'''

patientFolder_val = Path(r"/mnt/raid/work/data/DECT_data/validation")
if not server:
    patientFolder_val = Path(r"\home\cei\Codice\test2")

# Test path
'''patientFolder = Path(r"/home/rossetti/DECT_data/train")
if server == False:
    patientFolder = Path(r"D:\Tesi_magistrale\Codice\test2")'''

patientFolder_test = Path(r"/mnt/raid/work/data/DECT_data/test")
if not server:
    patientFolder_test = Path(r"/home/cei/Codice/test2")


data_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(data_dir, 'data')

# DA SCEGLIERE SE SALVARE DIRETTAMENTE IN data OPPURE SE LASCIARE IN expertiments/experiment_GG-MM-AAAA--OO-MM

# DataFrame of patient parameters (for patient by patient normalization)
# Genero lista dei paths delle immagini valide
if compute_dataset_paths:
    #Path dove vengono salvati i file
    datasetPaths_save = os.path.join(experimentPath, "dataset_paths.pkl")
    datasetPaths_val_save = os.path.join(experimentPath, "dataset_paths_val.pkl")
    datasetPaths_test_save = os.path.join(experimentPath, "dataset_paths_test.pkl")

    print("\nCalcolo dataset paths\n")
    DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath, AN_list_train = DatasetPaths(patientFolder)
    DSinputPath_val, DStargetPath_val, DSmaskPath_val, DSmaskRegPath_val, AN_list_val = DatasetPaths(patientFolder_val)
    DSinputPath_test, DStargetPath_test, DSmaskPath_test, DSmaskRegPath_test, AN_list_test = DatasetPaths(patientFolder_test)

    # Salva path delle immagini dei pazienti di training
    with open(datasetPaths_save, "wb") as f:
        pickle.dump((DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath, AN_list_train),f)

    # Salva path delle immagini dei pazienti di validation
    with open(datasetPaths_val_save, "wb") as f:
        pickle.dump((DSinputPath_val, DStargetPath_val, DSmaskPath_val, DSmaskRegPath_val, AN_list_val),f)

    # Salva path delle immagini dei pazienti di test
    with open(datasetPaths_test_save, "wb") as f:
        pickle.dump((DSinputPath_test, DStargetPath_test, DSmaskPath_test, DSmaskRegPath_test, AN_list_test),f)

# Carico lista dei paths delle immagini valide
else:
    # Path dei file da caricare, che sono situati nella cartella data
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


# Gestione dei parametri dei pazienti
if normalizeByPatient:

    # Calcolo e salvo su file i parametri dei pazienti
    if compute_patient_parameters:
        print('\nCalcolo parametri pazienti Train\n')
        df = patientsParameters(DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath, AN_list_train)
        patientParametersPath = os.path.join(experimentPath, "patientsParameters.pkl")
        df.to_pickle(patientParametersPath)

        print('\nCalcolo parametri pazienti Validation\n')
        df_val = patientsParameters(DSinputPath_val, DStargetPath_val, DSmaskPath_val, DSmaskRegPath_val, AN_list_val)
        patientParametersPath_val = os.path.join(experimentPath, "patientsParameters_val.pkl")
        df_val.to_pickle(patientParametersPath_val)

        print('\nCalcolo parametri pazienti Test\n')
        df_test = patientsParameters(DSinputPath_test, DStargetPath_test, DSmaskPath_test, DSmaskRegPath_test, AN_list_test)
        patientParametersPath_test = os.path.join(experimentPath, "patientsParameters_test.pkl")
        df_test.to_pickle(patientParametersPath_test)

    # Carico parametri pazienti invece di calcolarli
    else:

        # Path dei file da caricare, che sono situati nella cartella data
        patientParametersPath = os.path.join(data_dir, "patientsParameters.pkl")
        patientParametersPath_val = os.path.join(data_dir, "patientsParameters_val.pkl")
        patientParametersPath_test = os.path.join(data_dir, "patientsParameters_test.pkl")

        # Se i file non esistono ottengo errore specifico
        if not os.path.exists(patientParametersPath) or not os.path.exists(patientParametersPath_val) or not os.path.exists(patientParametersPath_test):
            raise FileNotFoundError(f"File dei parametri pazienti non trovati \n{patientParametersPath}\n{patientParametersPath_val}\n{patientParametersPath_test}\nEsegui prima con compute_patient_stats = True")


# Logging su file di errori fatali del codice
logging.basicConfig(
    filename=os.path.join(experimentPath, "crash.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

sys.excepthook = handle_exception



#%% GENERATOR creation

generator = Generator(n_canali=n_canali)


# experimentPath indica il path dei checkpoints che vogliamo usare
if test:
    # Versione batch 32 80 epoche lambda 700
    #exp = "experiment_27-04-2026--14-18" #Codice vecchio
    #exp = "experiment_11-05-2026--18-26" #Codice con ciclo train vecchio
    #exp = "experiment_09-05-2026--12-35"
    # Versione batch 8 100 epoche lambda 700
    #exp = "experiment_08-05-2026--12-59"
    # Versione batch 8 100 epoche lambda 100
    #exp = "experiment_07-05-2026--16-45"
    # Versione batch 8 400 epoche lambda 100
    exp = "experiment_14-05-2026--19-42"
    checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
    experimentPath_load = os.path.join(checkpoint_dir,"experiments", exp)

#%% Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Salvo checkpoint in cartella interna di experiment
checkpoint_prefix = os.path.join(experimentPath, "checkpoints", "cGANckpt.model.keras")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

#%% train
if train:
    # crea cartella che conterra' i plots delle loss
    log_dir = create_log_dir(experimentPath)
    
    #random seed generator for each array
    rng_input = np.random.default_rng(seed = 42)
    rng_target = np.random.default_rng(seed = 42)
    rng_mask = np.random.default_rng(seed = 42)
    rng_maskReg = np.random.default_rng(seed = 42)
    RNG_SHUFFLE = 42


# DataFrame of patient parameters (for patient by patient normalization)

if train:
    print('\nSTART TRAINING\n')
    # Define the output signature for the generator
    if n_canali == 1:
        output_signature = (
                tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32),  # Input image
                tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32)   # Label
            )
    else:
        output_signature = (
                tf.TensorSpec(shape=(512, 512, n_canali), dtype=tf.float32),  # Input images group
                tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32)   # Label
            )
    
    losses = {
          "gen_total",
          "gen_gan",
          "gen_l1",
          "disc_total",
          "disc_real",
          "disc_fake"
      }

    # Loss per tutti i batch di tutte le epoche
    losses_history = {
        "gen_total": [],
        "gen_gan": [],
        "gen_l1": [],
        "disc_total": [],
        "disc_real": [],
        "disc_fake": []
    }

    # Metriche delle loss per un'epoca
    losses_metrics = {
        "gen_total": tf.keras.metrics.Mean(),
        "gen_gan": tf.keras.metrics.Mean(),
        "gen_l1": tf.keras.metrics.Mean(),
        "disc_total": tf.keras.metrics.Mean(),
        "disc_real": tf.keras.metrics.Mean(),
        "disc_fake": tf.keras.metrics.Mean()
    }

    #Loss medie per tutte le epoche
    losses_history_epochs = {
        "gen_total": [],
        "gen_gan": [],
        "gen_l1": [],
        "disc_total": [],
        "disc_real": [],
        "disc_fake": []
    }

    f1_score_metrics = {
        "f1_gen": tf.keras.metrics.F1Score(average = "micro", threshold=0.5),
        "f1_real": tf.keras.metrics.F1Score(average = "micro", threshold=0.5)
    }

    f1_score_epochs = {
        "f1_gen": [],
        "f1_real": []
    }

    ## VALIDATION SET
    print('\nLoading Validation Dataset: \n')
    
    args_val = (DSinputPath_val, DStargetPath_val, DSmaskPath_val, DSmaskRegPath_val,
            normalizeByPatient, patientParametersPath_val, normalizationFunction, n_canali, stride)
    
    dataset_val = tf.data.Dataset.from_generator(DatasetGenerator,
                                             output_signature=output_signature,
                                             args = args_val).cache()
    
    args_train = (DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath,
                 normalizeByPatient, patientParametersPath, normalizationFunction, n_canali, stride)

    train_dataset = tf.data.Dataset.from_generator(DatasetGenerator,
                                                 output_signature=output_signature,
                                                 args = args_train).cache()
    
    val_dataset = dataset_val.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # DEBUG sul datasetGenerator
    '''a,b = DatasetGenerator(DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath,
                 normalizeByPatient, patientParametersPath, normalizationFunction, n_canali, stride)'''
    
    # DEBUG SHUFFLE
    '''args_val = (DSinputPath_val, DStargetPath_val, DSmaskPath_val, DSmaskRegPath_val,
            normalizeByPatient, patientParametersPath_val, normalizationFunction, n_canali, stride)
    
    output_signature = (
    tf.TensorSpec(shape=(), dtype=tf.string),
    tf.TensorSpec(shape=(), dtype=tf.string),
    )
    
    dataset_val = tf.data.Dataset.from_generator(DatasetGenerator_test,
                                                output_signature=output_signature,
                                                args = args_val).cache()
    
    args_train = (DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath,
                 normalizeByPatient, patientParametersPath, normalizationFunction, n_canali, stride)

    train_dataset = tf.data.Dataset.from_generator(DatasetGenerator_test,
                                                    output_signature=output_signature,
                                                    args = args_train).cache()
    
    val_dataset = dataset_val.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)'''
    
    print('\nLoading Train Dataset\n')
    #30GB di memoria
    train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality(), reshuffle_each_iteration=True, seed = RNG_SHUFFLE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    
    for epoch in range(epochs):

        print('\nEpoch ', epoch, '/', epochs,' \n')
      
        # Reset metriche per epoca attuale
        for key in losses:
          losses_metrics[key].reset_state()

        # Training vero e proprio della singola epoca
        losses_history, losses_metrics, f1_score_metrics = fit(train_dataset, val_dataset, _lambda, 
                                             generator, discriminator, generator_optimizer, discriminator_optimizer, 
                                             losses_history, losses_metrics, f1_score_metrics)

        # Ottengo valore medio epoca
        for key, metric in losses_metrics.items():
          losses_history_epochs[key].append(metric.result().numpy())

        for key, metric in f1_score_metrics.items():
          f1_score_epochs[key].append(metric.result().numpy())

        # Salvo modello ogni 15 epoche
        if epoch % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    # Salvo modello finale
    checkpoint.save(file_prefix=checkpoint_prefix)

    print('\nFINE TRAINING\n')

    print('\nGenerazione grafici losses del training \n')

    # Conversione da tensori a float
    losses_history_serializable = {
        k: [float(v_i.numpy()) for v_i in v]
        for k, v in losses_history.items()
    }

    losses_history_epochs_serializable = {
        k: [float(v_i) for v_i in v]
        for k, v in losses_history_epochs.items()
    }

    f1_score_epochs_serializable = {
        k: [float(v_i) for v_i in v]
        for k, v in f1_score_epochs.items()
    }

    # Salvo storico delle loss di ogni batch del training
    # Salvataggio usando json per leggibilita' migliore rispetto ad usare numpi.save()
    with open(os.path.join(experimentPath, "losses_batch.json"), "w") as f:
        json.dump(losses_history_serializable, f)

    # Salvo storico delle loss medie di ogni epoca del training
    with open(os.path.join(experimentPath, "losses_epochs.json"), "w") as f:
        json.dump(losses_history_epochs_serializable, f)

    # Salvo storico delle f1_score di ogni epoca del training
    with open(os.path.join(experimentPath, "f1_scores.json"), "w") as f:
        json.dump(f1_score_epochs_serializable, f)

    # GENERATOR tutte
    save_loss_plot(
        losses_history_epochs,
        ["gen_total", "gen_gan", "gen_l1"],
        "Generator Losses",
        "generator_losses.png",
        log_dir
    )

    # GENERATOR no total
    save_loss_plot(
        losses_history_epochs,
        ["gen_gan", "gen_l1"],
        "Generator Losses (no total)",
        "generator_losses_no_total.png",
        log_dir
    )

    # DISCRIMINATOR
    save_loss_plot(
        losses_history_epochs,
        ["disc_total", "disc_real", "disc_fake"],
        "Discriminator Losses",
        "discriminator_losses.png",
        log_dir
    )

    # DISCRIMINATOR e GENERATOR
    save_loss_plot(
        losses_history_epochs,
        ["gen_total", "disc_total"],
        "Discriminator and Generator Losses",
        "disc_gen_tot_losses.png",
        log_dir
    )

    # DISCRIMINATOR e GENERATOR loss della gan 
    save_loss_plot(
        losses_history_epochs,
        ["gen_gan", "disc_total"],
        "Discriminator and Generator GAN Losses",
        "disc_gen_gan_losses.png",
        log_dir
    )

    # DISCRIMINATOR e GENERATOR loss della gan 
    save_loss_plot_lambda(
        losses_history_epochs,
        ["gen_l1", "gen_gan"],
        "Generator L1*lambda e GAN Losses",
        "gen_l1_lambda_gan_losses.png",
        log_dir,
        _lambda = _lambda
    )

    # Andamento f1_score nel training
    save_loss_plot(
        f1_score_epochs,
        ["f1_gen", "f1_real"],
        "F1-score generated and real images",
        "f1_scores.png",
        log_dir
    )

    # Salvo file con tempo di esecuzione totale del training
    time_program = time.time()-start_program
    print(f'Tempo esecuzione totale programma: {time_program:.2f} sec\n')
    with open(os.path.join(experimentPath,"tempo_esecuzione.txt"), "w") as text_file:
        text_file.write("Tempo esecuzione totale programma: %s" % time_program)

#%% Loading Model    

if test:

    # Carico il modello salvato nella cartella dell'esperimento indicato sopra
    if exp == "experiment_27-04-2026--14-18":
        model = checkpoint.restore(tf.train.latest_checkpoint(experimentPath_load)) #Versione per esperimenti vecchi
    else: 
        model = checkpoint.restore(tf.train.latest_checkpoint(os.path.join(experimentPath_load, "checkpoints")))

    # Salvo storico delle loss di batch di ogni epoca del training
    with open(os.path.join(experimentPath_load, "losses_batch.json"), "r") as f:
        losses_history = json.load(f)

    # Salvo storico delle loss medie di ogni epoca del training
    with open(os.path.join(experimentPath_load, "losses_epochs.json"), "r") as f:
        losses_history_epochs = json.load(f)


#%% Plot 

# TEMP: MODIFICARE APPENA FANNO IN CIMA PER TEST
if test:

    # Scelta batch_size per visualizzazione
    batch_size = 32
    
    # Definizione della "firma" dell'output del generatore
    output_signature = (
        tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32),  # Input image
        tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32)   # Label
    )

    '''# DEBUG Definizione della firma per test
    output_signature = (
    tf.TensorSpec(shape=(), dtype=tf.string),
    tf.TensorSpec(shape=(), dtype=tf.string),
    )'''

    # Definizione dataset di test
    args_test = (DSinputPath_test, DStargetPath_test, DSmaskPath_test, DSmaskRegPath_test,
             normalizeByPatient, patientParametersPath_test, normalizationFunction, n_canali, stride)
        
    test_dataset = tf.data.Dataset.from_generator(DatasetGenerator,
                                             output_signature=output_signature,
                                             args = args_test)
    
    #batch the dataset
    test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

if test:

    # Creo cartela dove finiscono i plots, se già presente metto un 2 alla fine
    if os.path.exists(plotPath):       
        plotPath = plotPath + "2"
        
    os.mkdir(plotPath)

    with open(os.path.join(experimentPath,"modello_usato.txt"), "w") as text_file:
        text_file.write("Esperimento dal quale e' stato preso il modello: %s" % exp)
    
    # Carico dataset con parametri dei pazienti di test
    df = pd.read_pickle(patientParametersPath_test)

    # Lista dei pazienti di test utilizzati
    #['AN21661133', 'AN21846709', 'AN21016322', 'AN21598082', 'AN21565406']
    paz_path = df.get("AN").values.tolist()


    print("Pazienti usati nel test:\n", paz_path)

    # Definizione parametri per tipo di normalizzazione
    param_norm = {
        'minmax': ['input_min', 'input_max', 'target_min', 'target_max'],
        'standard': ['input_mean', 'input_std', 'target_mean', 'target_std']
    }

    # Controllo se la normalizzazione è già presente
    if normalizationFunction[0] not in param_norm:
        raise ValueError(f"Normalizzazione non supportata: {normalizationFunction[0]}")

    # Dict con struttura uguale a quella della normalizzazione
    results = {param: [] for param in param_norm[normalizationFunction[0]]}

    # Salvo parametri necessari ad invertire la normalizzazione
    for paz in paz_path:
        for param in param_norm[normalizationFunction[0]]:
            value = GetPatientParameters(paz, param, df)
            results[param].append(value)

    

    # DEBUG generazione immagini
    '''for step, (input_image, target) in enumerate(test_dataset):
  
      print(f"Batch: {step}")
      print("INPUT IMAGES:\n")
      for input_im in enumerate(input_image):
         print(input_im)
      
      print("TARGET IMAGES:\n")
      for target_im in enumerate(target):
         print(target_im)'''

    # Run the trained model on a few examples from the test set
    for i, (inp, tar) in enumerate(test_dataset.take(2)):
        #generate_images(generator, inp, tar, plot = True)
        pred = generator(inp, training=False)
        for ii, (inp_i, tar_i) in enumerate(zip(inp, tar)):

            if ii==11 or ii == 28:
                #fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10), sharex=True, sharey=True)
                #ax = axes.ravel()
                
                #inp = np.array(inp[2])
                #tar = np.array(tar[2])
                inp_ii = np.array(inp_i)
                tar_ii = np.array(tar_i)

                # TEST: controllare se dopo denormalizzazione tornano i valori del file .npy

                # CONTROLLARE COME DENORMALIZZARE OUTPUT (MEGLIO INPUT DATO CHE TEORICAMENTE TARGET NON LO ABBIAMO????)
                inp_scale, inp_shift = get_scale_shift(results, normalizationFunction[0], 'input', 0)
                tar_scale, tar_shift = get_scale_shift(results, normalizationFunction[0], 'target', 0)

                # Denormalizzo input, target e output (OUTPUT NORMALIZZATO CON DATI TARGET)
                inp_denorm  = inp_ii * inp_scale + inp_shift
                tar_denorm  = tar_ii * tar_scale + tar_shift
                pred_denorm = pred[ii] * tar_scale + tar_shift

                print(np.shape(inp_ii), np.shape(tar_ii), np.shape(pred[ii]))

                # DA CAMBIARE CON MASCHERA TARGET (MEGLIO INPUT DATO CHE TEORICAMENTE TARGET NON LO ABBIAMO????)
                pred_ = np.array(pred_denorm * np.where(inp_ii == 0, 0, 1))

                #TEST
                #pred_=pred_denorm
                
                # DEBUG generazione immagini
                #np.save(f"prova_pre_no_den_finestra_{i}.npy", tar_ii)
                #np.save(f"prova_pre_finestra_{i}.npy", tar_denorm)

                inp_window = apply_window(inp_denorm -1000, 40, 400)
                tar_window = apply_window(tar_denorm - 1000, 40, 400)
                pred_window = apply_window(pred_ - 1000, 40, 400)

                # DEBUG generazione immagini
                #np.save(f"prova_{i}.npy", tar_window)
                

                '''# Serviti per grafici vecchi?
                mse_input = mean_squared_error(tar_ii, inp_ii)
                ssim_input = skimage.metrics.structural_similarity(tar_ii, inp_ii, data_range=inp_ii.max() - inp_ii.min(), channel_axis = 2)
                
                mse_self = mean_squared_error(tar_ii, tar_ii)
                ssim_self = skimage.metrics.structural_similarity(tar_ii, tar_ii, data_range=tar_ii.max() - tar_ii.min(), channel_axis = 2)

                mse_output = mean_squared_error(tar_ii, pred_)
                ssim_output = skimage.metrics.structural_similarity(tar_ii, pred_, data_range=pred_.max() - pred_.min(), channel_axis = 2)'''

                plt.figure()
                plt.imshow(inp_denorm, cmap = 'gray')
                plt.axis('off')
                # Save the figure
                save_path = os.path.join(plotPath, f"_inp_image_{i}_{ii}.png")
                plt.savefig(save_path, dpi=300)
                print(f"Saved: {save_path}")
                
                plt.figure()
                plt.imshow(tar_denorm, cmap = 'gray')
                plt.axis('off')
                # Save the figure
                save_path = os.path.join(plotPath, f"_tar_image_{i}_{ii}.png")
                plt.savefig(save_path, dpi=300)
                print(f"Saved: {save_path}")
                
                plt.figure()
                plt.imshow(pred_, cmap = 'gray')
                plt.axis('off')
                # Save the figure
                save_path = os.path.join(plotPath, f"_out_image_{i}_{ii}.png")
                plt.savefig(save_path, dpi=300)
                print(f"Saved: {save_path}")

                plt.figure()
                plt.imshow(inp_window, cmap = 'gray')
                plt.axis('off')
                # Save the figure
                save_path = os.path.join(plotPath, f"_inp_image_window_{i}_{ii}.png")
                plt.savefig(save_path, dpi=300)
                print(f"Saved: {save_path}")
                
                plt.figure()
                plt.imshow(tar_window, cmap = 'gray')
                plt.axis('off')
                # Save the figure
                save_path = os.path.join(plotPath, f"_tar_image_window_{i}_{ii}.png")
                plt.savefig(save_path, dpi=300)
                print(f"Saved: {save_path}")
                
                plt.figure()
                plt.imshow(pred_window, cmap = 'gray')
                plt.axis('off')
                # Save the figure
                save_path = os.path.join(plotPath, f"_out_image_window_{i}_{ii}.png")
                plt.savefig(save_path, dpi=300)
                print(f"Saved: {save_path}")


