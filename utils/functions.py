import logging
import re
import sys

import pandas as pd
import numpy as np
from pydicom import dcmread
import os
import json
from scipy.ndimage import uniform_filter
from datetime import datetime

def minmax(img):
    out = (img - np.min(img))/(np.max(img) - np.min(img))
    return out

def standard(img):
    out = (img - np.mean(img))/(np.std(img) + 1e-8)
    return out

def normalization(data, normalizationFunction = ['standard']):
    
    # Lista opzioni di normalizzazione (aggiungibili)
    function_mappings = {
        'minmax': minmax,
        'standard': standard
        }
    
    for function_str in normalizationFunction:
        out = function_mappings[function_str](data)
    
    return out
    

def Image2Array(path):
    
    #Da Dicom a Numpy
    ds = dcmread(path)
    arr = ds.pixel_array

    return arr   
    

def SaveToFile(dictionary, filename, path):
    
    # Parameters to save
    
    '''
    params = {
        "x_mean": x_mean,
        "y_mean": y_mean,
        "x_std": x_std,
        "y_std": y_std,
    }
    '''
    
    # Save to a file
    with open(os.path.join(path,'{}.json'.format(str(filename))), 'w') as f:
        json.dump(dictionary, f)

def ANlist(patientFolder):
    return os.listdir(patientFolder)

def getPatientAN(imgPath):

    #imgPath full path to img
    fileName = os.path.basename(imgPath)
    # Prendo solamente il codice del paziente (AN########)
    # Va fatto controllo o non serve????
    AN = fileName.split('_')[0] 
    
    return AN


def GetPatientParameters(imgPath, parameter, df):
    
    AN = getPatientAN(imgPath)
    
    if parameter in df.columns:
        value = df.loc[df['AN'] == AN, str(parameter)].iloc[0]
    else:
        print('parametro non in elenco.')
    return value

def GetSpacing(image):
    
    ds = dcmread(image)

    spacing = ds['SliceThickness'].repval
      
    return spacing

# Controllo presenza di cluster di 0
def DiscardZerosImageCluster(image, mask, patch_size=10, var_threshold=1e-4, mean_threshold=5,):

    img = image.astype(np.float32)

    mean = uniform_filter(img, patch_size)
    mean_sq = uniform_filter(img**2, patch_size)

    variance = mean_sq - mean**2

    suspicious = (variance < var_threshold) & (np.abs(mean) < mean_threshold) & (mask == 1)

    if (np.any(suspicious)):
        print("trovato cluster sospetto, eliminato:")

    return np.any(suspicious)

# Controllo percentuale zeri della maschera
def DiscardZerosMasks(mask, max_perc_zero = 80):
    w = np.where((mask == 1), 1, 0)

    tot = w.shape[0] * w.shape[1]
    perc = (tot - np.count_nonzero(w)) / tot
    
    if perc*100 > max_perc_zero: 
        print("Percentuale zeri: ", perc)
        print("trovato maschera sospetta, eliminata:")
        return True
    
    return False

def datasetToNumpy(dataset):
    
    #data diventa una lista di batch, ha lunghezza N_samples/batch_size
    #ogni elemento è una tupla (batch_size, batch_size)
    data = list(dataset)
    
    x = []
    y = []
    
    for batch in data:
        x.append(batch[0])
        y.append(batch[1])
        
    #list of list to list
    X = [item for sublist in x for item in sublist]
    Y = [item for sublist in y for item in sublist]
    
    return np.array(X), np.array(Y)

def filter_inplace(lst, idx):
        lst[:] = [x for x, keep in zip(lst, idx) if keep]


def create_log_dir(base_path):
    log_dir = os.path.join(
        base_path,
        "plots_losses"
    )
    
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def crea_gruppi_path(lista_path, n_canali, stride=1):
    """
    Crea gruppi consecutivi (tecnica sliding window) di dimensione n_canali
    con stride a scelta.

    Esempio:
    lista = [1,2,3,4,5], n_canali=3

    stride=1 -> [[1,2,3], [2,3,4], [3,4,5]]
    stride=2 -> [[1,2,3], [3,4,5]]
    """
    if len(lista_path) < n_canali:
        return []

    gruppi = []
    for i in range(0, len(lista_path) - n_canali + 1, stride):
        gruppi.append(lista_path[i:i + n_canali])

    return gruppi

    #Versione compatta
    #return [lista[i:i+n] for i in range(0, len(lista)-n+1, stride)]

def normalize_by_patient(CT_input, CT_target, inputPath, targetPath, df, normalizationFunction):

    if normalizationFunction[0] == 'minmax':

        input_min = GetPatientParameters(inputPath, 'input_min', df)
        input_max = GetPatientParameters(inputPath, 'input_max', df)
        CT_input = (CT_input - input_min) / (input_max - input_min)

        target_min = GetPatientParameters(targetPath, 'target_min', df)
        target_max = GetPatientParameters(targetPath, 'target_max', df)
        CT_target = (CT_target - target_min) / (target_max - target_min)

    elif normalizationFunction[0] == 'standard':

        input_mean = GetPatientParameters(inputPath, 'input_mean', df)
        input_std = GetPatientParameters(inputPath, 'input_std', df)
        CT_input = (CT_input - input_mean) / (input_std + 1e-8)

        target_mean = GetPatientParameters(targetPath, 'target_mean', df)
        target_std = GetPatientParameters(targetPath, 'target_std', df)
        CT_target = (CT_target - target_mean) / (target_std + 1e-8)

    return CT_input, CT_target

def sort_path_key(path):
    head, tail = os.path.split(path)
    elem = tail.split("_")
    head_comp = head + "/" + elem[0] + "_" + elem[1]
    num = float(os.path.splitext(elem[2])[0])
    return (head_comp, -num)


'''# DEBUG per testare ordinamento
paths = ['/mnt/raid/work/data/DECT_data/train/AN20580638/registration/AN20580638_registered_101.750.npy', #3
         '/mnt/raid/work/data/DECT_data/train/AN20580638/registration/AN20580638_registered_-150.750.npy', #4
         '/mnt/raid/work/data/DECT_data/train/AN20580638/registration/AN20580638_registered_2.750.npy', #2
         '/mnt/raid/work/data/DECT_data/train/AN20580638/registration/AN20580638_registered_-10.750.npy', #5
         '/mnt/raid/work/data/DECT_data/train/AN10580638/registration/AN10580638_registered_-350.750.npy'] #1
head, tail = os.path.split(paths[0])

print("Head:", head)
print("Tail:", tail, "\n")


elem = tail.split("_")

head_comp = head + "/" + elem[0] + "_" + elem[1]
num = float(os.path.splitext(elem[-1])[0])

print("testa: ", head_comp)
print("numero: ", )

fileName = os.path.basename(paths[0])
# Prendo solamente il codice del paziente (AN########)
# Va fatto controllo o non serve????


ord = sorted(paths, key = sort_path_key)
print(ord)'''


def get_scale_shift(results, norm_type, kind, i, eps=1e-8):
    if norm_type == 'minmax':
        scale = results[f'{kind}_max'][i] - results[f'{kind}_min'][i]
        shift = results[f'{kind}_min'][i]
    elif norm_type == 'standard':
        scale = results[f'{kind}_std'][i] + eps
        shift = results[f'{kind}_mean'][i]
    else:
        raise ValueError(f"Normalizzazione non supportata: {norm_type}")
    
    return scale, shift

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error(
        "Eccezione non gestita",
        exc_info=(exc_type, exc_value, exc_traceback)
    )