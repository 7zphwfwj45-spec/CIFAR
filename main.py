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

lr = 0.0015
batchSize = 64
epochs = 3 
dropOutRate = 0.1
batchList = [[3]]          
loadParams = True

cifarInst = ci.cifar()
CNNManager = cnn.Control(batchSize, lr , epochs, dropOutRate)

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
print (" End of Training " +  " at " ,datetime.now().time())

"""""
runningCost = CNNManager.getCost()
plt.plot(runningCost)
plt.xlabel("Iterations")
plt.ylabel( "Error for all training instances")
plt.show()
plt.savefig("runningCost.png")  
"""""