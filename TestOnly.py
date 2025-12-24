import os
import numpy as np
import sys
from datetime import datetime
sys.path.append("/Volumes/MyDrive/sharedPythonFiles")
import CNN_Control as cnn
import cifar as ci


cifarInst = ci.cifar()

lr = 0.0
batchSize = 64
epochs = 1
dropOutRate = 0.0
         
print (" Start to Test at " ,datetime.now().time())

testImages, testLabels =cifarInst.getTestBatchForGPT()
CNNManager = cnn.Control(batchSize, lr , epochs, dropOutRate)
CNNManager.load_params()
CNNManager.test(testImages,testLabels)

print (" Finished Testing at " ,datetime.now().time())
        