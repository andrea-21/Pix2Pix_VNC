import numpy as np
import matplotlib.pyplot as plt
from functions import *

plt.figure()
#figura = np.load("/mnt/raid/work/data/DECT_data/train/AN20979285/registration/AN20979285_registered_-391.750.npy")
#maschera = np.load("/mnt/raid/work/data/DECT_data/train/AN20979285/registration_mask/AN20979285_registrered_-391.750_mask.npy")
test1 = np.load("/home/cei/codice/project/prova_0.npy")
plt.imshow(test1)
plt.savefig("immmagine1.png")
plt.figure()
test2 = np.load("/home/cei/codice/project/prova_1.npy")
plt.imshow(test2)
plt.savefig("immmagine2.png")
plt.figure()
test3 = np.load("/home/cei/codice/project/prova_2.npy")

#print(figura.min())
#print(np.where(figura == figura.min()))
#print(np.where(figura == 0))
#res =  DiscardZerosImageCluster(figura, maschera)
#plt.imshow(figura)
#plt.figure(2)
#plt.imshow(maschera)
#plt.figure(3)
plt.imshow(test3)
plt.axis('off')  # Turn off axis labels
plt.savefig("immmagine3.png")

test4 = np.load("/home/cei/codice/project/prova_pre_finestra_0.npy")
plt.imshow(test4)
plt.savefig("immmagine_pre1.png")
plt.figure()
test5 = np.load("/home/cei/codice/project/prova_pre_finestra_1.npy")
plt.imshow(test5)
plt.savefig("immmagine_pre2.png")
plt.figure()
test6 = np.load("/home/cei/codice/project/prova_pre_finestra_2.npy")
plt.imshow(test6)
plt.axis('off')  # Turn off axis labels
plt.savefig("immmagine_pre3.png")