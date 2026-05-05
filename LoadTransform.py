# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:51:58 2026

@author: PC
"""

import sys

import slicer

from DICOMLib import DICOMUtils
import os

import pydicom
from pydicom import dcmread
from pathlib import Path
import os
import numpy as np
import sys

import slicer
from DICOMLib import DICOMUtils
import shutil


print("entrato nel file")


def GetSlicePosition(image):
    
    ds = dcmread(image)

    z = round(ds['ImagePositionPatient'][-1], 3)         
      
    return z

def GetSeriesUID(folder):
    #Da Dicom a Numpy, mi salvo anche la posizione z del paziente
    file = os.listdir(folder)[0]

    filepath = os.path.join(folder, file)
    ds = dcmread(filepath) 

    seriesID = ds.SeriesInstanceUID
     
    return seriesID

def ImportSeries(folder):
    # instantiate a new DICOM browser
    slicer.util.selectModule("DICOM")
    dicomBrowser = slicer.modules.DICOMWidget.browserWidget.dicomBrowser
    # use dicomBrowser.ImportDirectoryCopy to make a copy of the files (useful for importing data from removable storage)
    dicomBrowser.importDirectory(folder, dicomBrowser.ImportDirectoryAddLink)
    # wait for import to finish before proceeding (optional, if removed then import runs in the background)
    dicomBrowser.waitForImportFinished()

def GetPatientAN(folder):
    
    file = os.listdir(folder)[0]
    filepath = os.path.join(folder, file)
    ds = dcmread(filepath)

    AN = ds.AccessionNumber
    #(0008,0050)

    return AN

def AutoMasking(folder):
    
    ImportSeries(folder)
    
    a = GetSeriesUID(folder)

    DICOMUtils.loadSeriesByUID([a])
    
    #list of nodes, i only have one
    node = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
    node = node[0]

    volume = slicer.util.arrayFromVolume(node)




    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation Mask")
     
       
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode) 
    segmentEditorWidget.setMasterVolumeNode(node) 


    mask = segmentationNode.GetSegmentation().AddEmptySegment('mask', "", [1.0,1.0,0.0])

    segmentEditorNode.SetSelectedSegmentID(mask)
    segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)
    segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)

    #THRESHOLD
    segmentEditorWidget.setActiveEffectByName("Threshold")

    min_t = np.min(volume)

    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MinimumThreshold",str(min_t))
    effect.setParameter("MaximumThreshold",str(-250))  
    effect.self().onApply()


    #CLOSING

    segmentEditorNode.SetSelectedSegmentID(mask)
    segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)
    segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)

    segmentEditorWidget.setActiveEffectByName("Smoothing")    
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("SmoothingMethod","MORPHOLOGICAL_CLOSING")
    effect.setParameter("KernelSizeMm",str(6))  
    effect.self().onApply()   

    '''
    #OPENING
    #per levare connessioni fatte dal closing

    segmentEditorNode.SetSelectedSegmentID(mask)
    segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)
    segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)

    segmentEditorWidget.setActiveEffectByName("Smoothing")    
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("SmoothingMethod","MORPHOLOGICAL_OPENING")
    effect.setParameter("KernelSizeMm",str(12))  #metto un pizzico in più
    effect.self().onApply()  

    '''

    #KEEP LARGEST ISLAND

    segmentEditorNode.SetSelectedSegmentID(mask)
    segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)
    segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)

    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation","KEEP_LARGEST_ISLAND")
    #effect.setParameter("BypassMasking","0")  
    effect.self().onApply()
        


    #INVERT

    segmentEditorNode.SetSelectedSegmentID(mask)
    segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)
    segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)


    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation","INVERT")
    #effect.setParameter("ModifierSegmentID", PA)
    effect.self().onApply()  


    segmentEditorWidget.setActiveEffectByName("None")
    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode) 


    #get array from segment

    #mask_array = slicer.util.arrayFromSegment(segmentationNode, mask)
    '''
    #salvo nella cartella del paziente con il nome della cartella di riferimento, esempio reg_mask
    parent_path = os.path.dirname(folder)
    dirname = os.path.basename(folder)
    save_path = os.path.join(parent_path, "{}_mask".format(dirname))
    np.save(save_path, mask_array)
    '''
    
    '''
    parent_path = os.path.dirname(folder)
    folder_name = os.path.basename(folder)
    #name = folder_name + "_mask"
    if 'VMI' in folder_name:
        name = 'VMI_mask'
        seriesType = 'VMI'
    elif 'TUE' in folder_name:
        name = 'TUE_mask'
        seriesType = 'TUE'
    
    mask_path = os.path.join(parent_path, name)
    
    
    
    
    AN = GetPatientAN(folder)
    
    
    z = []
    for img in os.listdir(folder):
        z.append(GetSlicePosition(os.path.join(folder,img)))
        
    z = sorted(z) #files are not ordered inside the folder, hence the need of sorting the z
        
    for ix, img in enumerate(mask_array):
        imgName = AN + '_' + seriesType + "_{}".format("%.3f"%z[ix]) + '_mask'
        save_path = os.path.join(mask_path, imgName)
        np.save(save_path, img)
    '''    

VUE_path = os.path.join(sys.argv[1], "VUE")

AutoMasking(VUE_path)



def ImportTransform(filepath):
    import slicer
    
    transformNode = slicer.util.loadTransform(filepath)
    
    if not transformNode:
        raise RuntimeError(f"Failed to load transform: {filepath}")
    
    return transformNode

transformPath = os.path.join(sys.argv[1], "VUE_transform.h5")
print(transformPath)

transform = ImportTransform(transformPath)


# Load Slicer nodes for segmentation and transform
segmentationNode = slicer.util.getNode('Segmentation Mask')
transform = slicer.util.getNode("VUE_transform")


# Apply transform
segmentationNode.SetAndObserveTransformNodeID(transform.GetID())

# Harden (apply)
slicer.vtkSlicerTransformLogic().hardenTransform(segmentationNode)





#


def CopyMask(folder):
    
    TUE_path = os.path.join(folder, "TUE")
    
    ImportSeries(TUE_path)
    
    a = GetSeriesUID(TUE_path)
    
    DICOMUtils.loadSeriesByUID([a])
    
    #list of nodes, i only have one
    volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")

    filteredVolumes = [
        node for node in volumeNodes
        if "soppresso" not in node.GetName().lower()
    ]
    node = filteredVolumes[0]
    
    volume = slicer.util.arrayFromVolume(node)
    
    
    segmentationNode = slicer.util.getNode('Segmentation Mask')
    mask_old = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('mask') 

    segmentationNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation Mask 2")
    segmentationNode2.CreateDefaultDisplayNodes() # only needed for display
    segmentationNode2.SetReferenceImageGeometryParameterFromVolumeNode(node)
    segmentationNode2.GetSegmentation().CopySegmentFromSegmentation(segmentationNode.GetSegmentation(), mask_old,
                                                                 False)
    
    '''
    PE = segmentationPE.GetSegmentIdBySegmentName('PE') #prima PE_II
    
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Metrica Embolie")
    segmentationNode.CreateDefaultDisplayNodes() # only needed for display
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
    
    segmentationNode.GetSegmentation().CopySegmentFromSegmentation(segmentationPE, PE,
                                                                     False)
    
    '''
    
    '''
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode) 
    segmentEditorWidget.setMasterVolumeNode(node) 
    
    
    mask = segmentationNode.GetSegmentation().AddEmptySegment('mask', "", [0.0,1.0,1.0])
    
    segmentEditorNode.SetSelectedSegmentID(mask)
    segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)
    segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
    
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation","COPY")
    effect.setParameter("ModifierSegmentID", mask)
    effect.self().onApply()  
    '''
    
CopyMask(sys.argv[1])  

      
# get segment and segment to array
segmentationNode = slicer.util.getNode('Segmentation Mask 2')
mask = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('mask')
mask_array = slicer.util.arrayFromSegment(segmentationNode, mask)

import scipy

filled_mask = []

for single_mask in mask_array:
    filled_mask.append(scipy.ndimage.binary_fill_holes(single_mask))

mask_array = filled_mask

'''
#salvo nella cartella del paziente con il nome della cartella di riferimento, esempio reg_mask
parent_path = os.path.dirname(folder)
dirname = os.path.basename(folder)
save_path = os.path.join(parent_path, "{}_mask".format(dirname))
np.save(save_path, mask_array)
'''

#path to mask folder
reg_mask = os.path.join(sys.argv[1], "registration_mask")
TUE_path = os.path.join(sys.argv[1], "TUE")

if not os.path.exists(reg_mask):       
    os.mkdir(reg_mask)


AN = GetPatientAN(VUE_path)

# list of z based on the VUE dicom images
z = []
for img in os.listdir(TUE_path):
    z.append(GetSlicePosition(os.path.join(TUE_path,img)))
    
z = sorted(z) #files are not ordered inside the folder, hence the need of sorting the z

print("lunghezza z", len(z))

# img are z slices in npy mask_array

for ix, img in enumerate(mask_array):
    
    imgName = AN + "_registered_{}".format("%.3f"%z[ix]) + '_mask'
    save_path = os.path.join(reg_mask, imgName)
    np.save(save_path, img)


sys.exit(0)









