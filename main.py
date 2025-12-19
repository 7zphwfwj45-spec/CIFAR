import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
sys.path.append("/Volumes/MyDrive/sharedPythonFiles")
import NeuralNetDefinition as nt          
import MatrixScaling as ms
import CNN_Control as cnn

import cifar as ci

print ("*-------------------------------CNN-CIFAR---------------------------------------------------*/")
cifarInst = ci.cifar()
lr = 0.004
batchSize = 64
epochs = 5
dropOutRate = 0.1

batchList = [[0]]           
loadParams = False

testImages, testLabels =cifarInst.getTestBatchForGPT()

# lr is saved with the "save_params_by_name" call 

CNNManager = cnn.Control(batchSize, lr , epochs, dropOutRate)

# save parms after testing each batch
print (" Start to Train " +  " at " ,datetime.now().time())
for ii in range(len(batchList)):  
    trainingImages, trainingLabels =cifarInst.getBatchDataFromListGPT(batchList[ii])
    CNNManager.setTrainingData(trainingImages, trainingLabels)
    
    if  loadParams == True:
        CNNManager.load_params()
        CNNManager.setLearningRate(lr)
        loadParams = False
    else:
        if ii == 0:   
            CNNManager.init()
            
    CNNManager.run()
    
CNNManager.save_params_by_name("model.pkl")
 
print (" Start to Test at " ,datetime.now().time())
CNNManager.setDropout (0.0)
CNNManager.test(testImages,testLabels)
CNNManager.setDropout (dropOutRate)
print (" Finished Testing at " ,datetime.now().time())
    

     