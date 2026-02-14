import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
sys.path.append("/Volumes/MyDrive/sharedPythonFiles")
import CNN_Control as cnn

import cifar as ci

print ("*-------------------------------CNN-CIFAR---------------------------------------------------*/")

cifarInst = ci.cifar()
images, labels = cifarInst.getManyBatches ([0,1,2,3,4 ])


lr = 0.001
batchSize = 64
epochs = 1
dropOutRate = 0.0
weightDecay = 5e-4
loadSaved = False
gapInUse = True
convLayers = [
            {'filter_size': 3, 'in_ch': 3, 'out_ch': 32, 'stride': 1, 'pad': 1, 'use_bn': True,
            'pool_param': {'pool_height': 0, 'pool_width': 0, 'stride': 0}},
            {'filter_size': 3, 'in_ch': 32, 'out_ch': 32, 'stride': 1, 'pad': 1, 'use_bn': True,
            'pool_param': {'pool_height': 2, 'pool_width': 2, 'stride': 2}},
            {'filter_size': 3, 'in_ch': 32, 'out_ch': 64, 'stride': 1, 'pad': 1, 'use_bn': True,
            'pool_param': {'pool_height': 0, 'pool_width': 0, 'stride': 0}},
            {'filter_size': 3, 'in_ch': 64, 'out_ch': 64, 'stride': 1, 'pad': 1, 'use_bn': True,
            'pool_param': {'pool_height': 2, 'pool_width': 2, 'stride': 2}},
            {'filter_size': 3, 'in_ch': 64, 'out_ch': 128, 'stride': 1, 'pad': 1, 'use_bn': True,
            'pool_param': {'pool_height': 0, 'pool_width': 0, 'stride': 0}},
            {'filter_size': 3, 'in_ch': 128, 'out_ch': 128, 'stride': 1, 'pad': 1, 'use_bn': True,
            'pool_param': {'pool_height': 2, 'pool_width': 2, 'stride': 2}}

        ]
denseChannels =  [10]


CNNManager = cnn.Control(batchSize, lr , epochs, dropOutRate, weightDecay, gapInUse)
CNNManager.setAugmentation(True)
CNNManager.setTrainingData(images,labels)
CNNManager.setConvLayers(convLayers)
CNNManager.setDenseChannels(denseChannels)


if loadSaved == True:
    CNNManager.load_params()
else:
    CNNManager.init()
    
CNNManager.printMessage (" Started to Train at ")
CNNManager.run()
CNNManager.printMessage (" Finished Training at ") 

runningCost = CNNManager.getCost()


CNNManager.save_params_by_name("model.pkl")

CNNManager.testTraining()
CNNManager.printMessage (" Finished Test Training at ")
print (" Start to Test at " ,datetime.now().time())
testImages, testLabels =cifarInst.getTestBatchForGPT()
CNNManager.test(testImages,testLabels)

print (" Finished Testing at " ,datetime.now().time())

plt.plot(runningCost)
plt.xlabel("Iterations")
plt.ylabel( "Error for all training instances")
plt.savefig("runningCost.png")
plt.show()