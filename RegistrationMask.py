# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:09:25 2026

@author: PC
"""

import pydicom
from pydicom import dcmread
from pathlib import Path
import os
import numpy as np
import sys

import shutil


#inputFolder = r"C:\PhD\DECT\test2"
inputFolder = "D:\\Tesi_magistrale\\Codice\\test2"


'''for patient in os.listdir(inputFolder):
    patientFolder =  os.path.join(inputFolder, patient)
    print(patientFolder)
    #3D Slicer 5.10.0
    #os.system('""D:\\Slicer 5.6.1\\Slicer.exe" --python-script "D:\\Tesi_magistrale\\Codice\\LoadTransform.py" {}"'.format(patientFolder))
    #os.system('"D:\\Slicer 5.6.1\\Slicer.exe" --python-script ''"D:\\Tesi_magistrale\\Codice\\LoadTransform.py" ''"{}"'.format(patientFolder))'''
    
import subprocess

slicer = r"D:\Slicer_5.6.1\Slicer.exe"
script = r"D:\Tesi_magistrale\Codice\project\LoadTransform.py"

for patient in os.listdir(inputFolder):
    patientFolder = os.path.join(inputFolder, patient)
    print(patientFolder)

    subprocess.run([
        slicer,
        "--python-script",
        script,
        patientFolder
    ])

#os.system('""D:\\Slicer 5.6.1\\Slicer.exe" --python-script r"D:\\Tesi_magistrale\\Codice\\LoadTransform.py" r"D:\\Tesi_magistrale\\Codice\\test2\AN21013671""')