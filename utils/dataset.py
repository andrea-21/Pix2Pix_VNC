import gc
import os

import numpy as np
import tensorflow as tf

from utils.functions import *

VALORE_MIN = 1024

def patientsParameters(DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath, AN_Database):    
    
    df = None

    # set di elementi da rimuovere globalmente
    to_remove = set()

    for AN in AN_Database:
        
        patientInputImgs = [s for s in DSinputPath if AN in s]
        patientTargetImgs = [s for s in DStargetPath if AN in s]

        patientInputImgs_mask = [s for s in DSmaskRegPath if AN in s]
        patientTargetImgs_mask = [s for s in DSmaskPath if AN in s]

        
        print('\n Calcolo Parametri', AN, '\n')
        inputCT = []
        targetCT = []

        for inputImg, targetImg, target_mask, input_mask in zip(patientInputImgs, patientTargetImgs, patientTargetImgs_mask, patientInputImgs_mask):

            
            inputCT_mask = np.load(input_mask)
            inputCT_single = np.load(inputImg)

            # Controllo presenza cluster di 0
            discard_cluster = DiscardZerosImageCluster(inputCT_single, inputCT_mask)
        
            # Controllo percentuale zeri maschera
            discard_mask = DiscardZerosMasks(inputCT_mask)
            
            # Almeno 1 tra le due opzioni basta per eliminare (non carico la sezione/maschera) 
            if discard_cluster or discard_mask:
                print("input_path:", inputImg)
                print("mask_path:", input_mask)
                print("discard cluster: ", discard_cluster)
                print("discard masks: ", discard_mask)
                print("ELMINATED IMAGE:", os.path.basename(inputImg))
                print("\n")
                # segna per rimozione globale
                to_remove.add((inputImg, targetImg, target_mask, input_mask))
                continue

            # Ritagio con maschera
            # Riporto valori in range [0, 4096] (outliers presenti nel file .npy, presenti sul bordo tra ct sferica e vuoto estero)
            inputCT_single = (np.load(inputImg) + VALORE_MIN) * inputCT_mask

            inputCT.append(inputCT_single)

            # Ritagio con maschera
            targetCT.append(Image2Array(targetImg) * np.load(target_mask))

        # SISTEMO PUNTI ANOMALI < 0
        inputCT = np.array(inputCT)
        # ATTENSIONE FORSE PROBLEMA DA CONVERSIONE IN np.array !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        inputCT[inputCT < 0] = 0

        patient_df = pd.DataFrame({
            'AN': [AN],
            'input_mean': [np.mean(inputCT)],
            'input_std': [np.std(inputCT)],
            'input_max': [np.max(inputCT)],
            'input_min': [np.min(inputCT)],

            'target_mean': [np.mean(targetCT)],
            'target_std': [np.std(targetCT)],
            'target_max': [np.max(targetCT)],
            'target_min': [np.min(targetCT)]
        })
        
        
        if df is not None:
            #metto prima la concatenazione, altrimenti mi concatena lui stesso
            #dopo il prossimo if statement
            df = pd.concat([df, patient_df], ignore_index = True)
            
        elif df is None:
            #assegno il dataframe al primo paziente
            df = patient_df

    # Modifica IN-PLACE delle liste originali
    keep_mask = []
    for items in zip(DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath):
        keep_mask.append(items not in to_remove)

    filter_inplace(DSinputPath, keep_mask)
    filter_inplace(DStargetPath, keep_mask)
    filter_inplace(DSmaskPath, keep_mask)
    filter_inplace(DSmaskRegPath, keep_mask)
        
    return df

def DatasetPaths(patientFolder):    
    
    DSinputPath = []
    DStargetPath = []    
    DSmaskPath = []
    DSmaskRegPath = []
    AN_list = []

    for patient in os.listdir(patientFolder):
        print('Loading path of patient: ', patient)
        patientPath = os.path.join(patientFolder, patient)
        
        folder_reg = os.path.join(patientPath, 'registration')
        folder_TUE = os.path.join(patientPath, 'TUE')
        folder_reg_mask = os.path.join(patientPath, 'registration_mask')
        folder_TUE_mask = os.path.join(patientPath, 'TUE_mask')

        skip = False

        for folder in [folder_reg, folder_reg_mask, folder_TUE, folder_TUE_mask]:

            # Se paziente non ha una tra TUE o TUE_mask
            if not os.path.exists(folder):
                #se ho un paziente che non ha folder_target
                print('missing ', folder.split('/')[-1] , 'folder... skipped\n')
                skip = True
                break
            elif not os.listdir(folder):
                #se ho un paziente che non folder_target vuota
                print(folder.split('/')[-1] ,  'folder empty... skipped\n')
                skip = True
                break
        # Se ha cartelle vuote o mancanti salta al prossimo
        if skip:
            continue
        
        TUE_files = os.listdir(folder_TUE) #sarebbe la GT
        TUE_files = np.array(TUE_files)
        
        
        image = TUE_files[0] #any image
        imagePath = os.path.join(folder_TUE, image)
        spacing_TUE = GetSpacing(imagePath)
        spacing_TUE = float(spacing_TUE.strip("'"))

        #print(spacing_TUE)
        
        image = os.listdir(os.path.join(patientPath, 'VUE'))[0] #any image
        imagePath = os.path.join(os.path.join(patientPath, 'VUE'), image)
        spacing_VUE = GetSpacing(imagePath)
        spacing_VUE = float(spacing_VUE.strip("'"))
        
        print('VUE:', spacing_VUE, 'TUE:', spacing_TUE)
        
        if spacing_TUE != 2.5:
            print('Spacing not matching... skipped\n')
            #check rounding error
            continue

        # Aggiungo alla lista dei pazienti
        AN_list.append(patient)
                
        #patient_pair = []
        
        for file in os.listdir(folder_reg):
            AN = file.split("_registered_",1)[0] 
            z_string = file.split('_registered_',1)[1]
            z_VUE = z_string.replace('.npy', '')
            
            #TUE_string = file.split("TUE_",1)[1]
            #matching_index = np.where(np.char.find(TUE_files, z_VUE) > -1)[0]
            fileName_TUE =  AN + '_TUE_' + z_VUE 
            fileName_reg =  AN + '_registered_' + z_VUE
            
            
            #CERCO 120 CHE MATCHANO
            matching_index = np.where(fileName_TUE == TUE_files)[0][0]
            
            #folder of binary masks
            maskPath = os.path.join(patientPath, 'TUE_mask') 
            maskRegPath = os.path.join(patientPath, 'registration_mask')
            
            #path to binary mask image, and loading
            #maskPath = os.path.join(patientPath, 'TUE_mask')
            maskPath = os.path.join(maskPath, fileName_TUE + '_mask.npy')
            
            # FIX MASCHERA REG PATH

            #path to binary mask image, and loading
            maskRegPath = os.path.join(maskRegPath, fileName_reg + '_mask.npy')
            
            #target
            CT_targetPath = os.path.join(folder_TUE, TUE_files[matching_index])

            #input
            CT_inputPath = os.path.join(folder_reg, file)
            
            
            DSinputPath.append(CT_inputPath)
            DStargetPath.append(CT_targetPath)
            DSmaskPath.append(maskPath)
            DSmaskRegPath.append(maskRegPath)
                
            #tupla di path che matchano
            #patient_pair.append((CT_targetPath, CT_inputPath))
            

    return DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath, AN_list

# Versione del generatore da usare con 2.5D
def DatasetGenerator(DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath,
                     normalizeByPatient, patientParametersPath, normalizationFunction, n_canali, stride):
    
    print('\n')
    print('normalize by patient: ', normalizeByPatient)
    print('\n')
    if n_canali == 1:
        print('Versione standard Dataset Generator start\n')
    else:
        print("Versione 2.5D Dataset Generator start\n")


    if normalizeByPatient == True:
        df = pd.read_pickle(patientParametersPath.decode('utf-8'))
    
    #implicit conversion! list > np, str > bytes!
    DSinputPath = [x.decode('utf-8') for x in DSinputPath]
    DStargetPath = [x.decode('utf-8') for x in DStargetPath]
    DSmaskPath = [x.decode('utf-8') for x in DSmaskPath]
    DSmaskRegPath = [x.decode('utf-8') for x in DSmaskRegPath]

    normalizationFunction = [x.decode('utf-8') for x in normalizationFunction]

    offset = n_canali // 2
    
    # Generatore nel caso 2D classico
    if n_canali == 1:
        
        for inputPath, targetPath, maskTargetPath, maskInputPath in zip(DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath):

            # Carico mashere per input e traget
            mask_input = np.load(maskInputPath)
            mask_target = np.load(maskTargetPath)

            # Carico immagine di input
            # Riporto valori in range [0, 4096] (outliers presenti nel file .npy, presenti sul bordo tra ct sferica e vuoto estero)
            CT_input = np.load(inputPath)
            CT_input = CT_input + VALORE_MIN

            CT_target = Image2Array(targetPath)

            # Normalizzazione per paziente
            if normalizeByPatient:

                CT_input, CT_target = normalize_by_patient(CT_input, CT_target, inputPath, targetPath, df, normalizationFunction)

            # Apllica maschera a input e target
            CT_input = CT_input * mask_input
            CT_target = CT_target * mask_target

            # Modifica formato del target e dell'input per farlo corrispondere a quello di tf
            input_img = np.expand_dims(CT_input, axis=-1)
            target_img = np.expand_dims(CT_target, axis=-1)

            # USO float32???
            #yield input_img.astype(np.float32), target_img.astype(np.float32)
            yield input_img, target_img

        # Caso generatore per 2.5D
    else:
        # Lista slice ordinate
        slices_input = sorted(DSinputPath, key=sort_path_key)
        slices_target = sorted(DStargetPath, key=sort_path_key)
        slices_mask_target = sorted(DSmaskRegPath, key=sort_path_key)
        slices_mask_input = sorted(DSmaskPath, key=sort_path_key)

        #print(slices_input)

        gruppi_input = crea_gruppi_path(slices_input, n_canali, stride)
        gruppi_target = crea_gruppi_path(slices_target, n_canali, stride)
        gruppi_mask_input = crea_gruppi_path(slices_mask_input, n_canali, stride)
        gruppi_mask_target = crea_gruppi_path(slices_mask_target, n_canali, stride)

        for gruppo_input, gruppo_target, gruppo_mask_input, gruppo_mask_target in zip(gruppi_input, gruppi_target, gruppi_mask_input, gruppi_mask_target):

            # Input 2.5D
            # Riporto valori in range [0, 4096] (outliers presenti nel file .npy, presenti sul bordo tra ct sferica e vuoto estero)
            input_stack = [(np.load(p) + VALORE_MIN) for p in gruppo_input]
            CT_input = np.stack(input_stack, axis=-1)  # (H, W, n_canali)

            # Maschera input
            mask_stack = [np.load(p) for p in gruppo_mask_input]
            mask_input = np.stack(mask_stack, axis=-1)

            CT_target = Image2Array(gruppo_target[offset])

            # Machera target centrale
            mask_target = np.load(gruppo_mask_target[offset])

            # Normalizzazione per paziente
            if normalizeByPatient:

                center_input_path = gruppo_input[offset]
                center_target_path = gruppo_target[offset]

                CT_input, CT_target = normalize_by_patient(CT_input, CT_target, center_input_path, center_target_path, df, normalizationFunction)

            # Apllica maschera a input e target
            CT_input = CT_input * mask_input
            CT_target = CT_target * mask_target

            # Modifica formato del target per farlo corrispondere a quello di tf
            CT_target = np.expand_dims(CT_target, axis=-1)

            # USO float32???
            #yield CT_input.astype(np.float32), CT_target.astype(np.float32)
            yield CT_input, CT_target

    gc.collect()


'''# Versione "classica" del generatore, da usare quando non si vuole il 2.5D
def DatasetGenerator(DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath,
                     normalizeByPatient, patientParametersPath, normalizationFunction):
    
    print('\n')
    print('normalize by patient: ', normalizeByPatient)
    print('\n')
    print('Dataset Generator start\n')
    
    if normalizeByPatient == True:
        df = pd.read_pickle(patientParametersPath.decode('utf-8'))
    
    #implicit conversion! list > np, str > bytes!
    DSinputPath = [x.decode('utf-8') for x in DSinputPath]
    DStargetPath = [x.decode('utf-8') for x in DStargetPath]
    DSmaskPath = [x.decode('utf-8') for x in DSmaskPath]
    DSmaskRegPath = [x.decode('utf-8') for x in DSmaskRegPath]

    normalizationFunction = [x.decode('utf-8') for x in normalizationFunction]

    
    for inputPath, targetPath, maskTargetPath, maskInputPath in zip(DSinputPath, DStargetPath, DSmaskPath, DSmaskRegPath):

        #Load mask
        mask_input = np.load(maskInputPath)
        mask_target = np.load(maskTargetPath)
        
        #Load input
        CT_input = np.load(inputPath)
        
        # Normalizzo portando valori sezione nel range [0, 4096] invece di [-1024, 3072]
        CT_input = CT_input + abs(CT_input.min())

        #Load target
        CT_target = Image2Array(targetPath)

        # Normalization image by image
        if normalizeByPatient == False:
           
            CT_input = CT_input * mask_input            
            CT_target = CT_target * mask_target

            
        # Normalization patient by patient
        if normalizeByPatient == True:
            
            if normalizationFunction[0] == 'minmax':
                
                #input
                input_min = GetPatientParameters(inputPath, 'input_min', df)
                input_max = GetPatientParameters(inputPath, 'input_max', df)
                
                CT_input = (CT_input - input_min)/(input_max - input_min)

                #target
                target_min = GetPatientParameters(targetPath, 'target_min', df)
                target_max = GetPatientParameters(targetPath, 'target_max', df)
                
                CT_target = (CT_target - target_min)/(target_max - target_min)
                
                
            if normalizationFunction[0] == 'standard':
  
                #input
                input_mean = GetPatientParameters(inputPath, 'input_mean', df)
                input_std = GetPatientParameters(inputPath, 'input_std', df)
                
                CT_input = (CT_input - input_mean)/(input_std + 1e-8)
                
                
                #target
                target_mean = GetPatientParameters(targetPath, 'target_mean', df)
                target_std = GetPatientParameters(targetPath, 'target_std', df)
                
                CT_target = (CT_target - target_mean)/(target_std + 1e-8)            

        CT_input = CT_input * mask_input
        CT_target = CT_target * mask_target

        #expand dims 
        input_img = np.expand_dims(CT_input, axis=-1)
        target_img = np.expand_dims(CT_target, axis=-1)


        yield input_img, target_img

    
    print('Dataset Generator end\n')

    gc.collect()'''
