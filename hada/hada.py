import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from tensorflow import keras
from PIL import Image as im


def RedBackground (Xc):
    X = Xc.copy()   
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            if (X[j,k,:] < [.12,.12,.12]).all():
                X[j,k,:] = [1,.2,.2]
            elif (X[j,k,:] > [.3,.3,.3]).all():
                X[j,k,:] = [0,0,0]
    return X

def EdgeHelper (Xc):
    X = Xc.copy()
    Y = cv2.Canny(X[0],.4,.8) #canney edge detecton
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] == 1:
                Y[0][i,j,:] = [.2,.05,.05]  
    return Y  

def RandomRotate(Xc):
    X = Xc.copy()
    height, width = X.shape[1:3]
    center = (width/2, height/2)
    for i in range(X.shape[0]):
        rand = random.randint(0,360)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rand, scale=1)
        rotated_image = cv2.warpAffine(src=X[i], M=rotate_matrix, dsize=(width, height))
        X[i] = rotated_image
    return X

def Resize(Xc):
    X = Xc.copy()
    arr = cv2.resize(X,(X.shape[1],X.shape[2]))
    return arr

def Modify (X):
    Xa = X.copy()
    Xb = RandomRotate(Xa)
    #Xd = EdgeHelper(Xc)
    Xc = RedBackground(Xb)
    return Xc

