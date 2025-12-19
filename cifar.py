import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import showImage
import sys
import tensorflow as tf

from tensorflow import keras
import pickle
from tensorflow.keras.models import Sequential
# nore Conv2D is for imaged and Conv3D is for video
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar10
import MatrixScaling as ms



from keras.datasets import cifar10

# THis works ok but the downside is that you have to set path in each python files that needs the module
sys.path.append("/Volumes/MyDrive/sharedPythonFiles")

# There are five batches that contain the self.training data 50K images and labels
# theee is a one batch of self.test data containing 10K imaghes and 10K labels

# The pixels are stored as unsigned 8 bit integers so convert to float32

class cifar:
    def __init__(self):
        self.trainingImages = np.zeros((10000,32,32,3))
        self.trainingLabels = np.zeros((10000,10))
        self.testImages = np.zeros((1000,32,32,3))
        self.testLabels = np.zeros((1000,10))
        self.batchNames = ['data_batch_1', 'data_batch_2','data_batch_3','data_batch_4','data_batch_5', ]
        print ("Hello From CIFAR-10")
        self.showImage = False
        # that of the last image displayed.
        self.indexOfDisplayedImage = -1
        self.displayedImageType = -1
        # aways return one image type
        self.nImages = 1
        
        self. imageTypes = ["airplane","automobile", "bird","cat","deer","dog","frog","horse","ship","truck"]

    
    def showRandomImage(self, status, num):
        self.showImage = status
        self.nImages = num
    
    def extractDataFromBatch(self, batch):
        with open(batch, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        keys = dict.keys()
        keysList = list(keys)
        print ("Batch Name is ", batch)
        
        labelsDict = dict[keysList[1]]
        labels = np.zeros((len(labelsDict),1))
        labels[:,0 ] = labelsDict
        imagesDict = dict[keysList[2]]
        images = imagesDict
        
        return images, labels
    
    def getBatchData(self, batch):
        
        images,labels = self.extractDataFromBatch(batch)
        # the rehape convert images into a fout dimensionsla shape .
        # transpose moves the axis to give a shape (10000,32,32,3)
        imagesR = np.reshape(images, (10000,3,32,32)).transpose(0,2,3,1).astype('float32')
        labels = to_categorical(labels)
    
#  display before normalizing.
        for ii in range(self.nImages):
            self.indexOfDisplayedImage = self.displayOneSample(imagesR, labels)
            
        return imagesR/255, labels
    
    def getBatchDataFromBatchNumber (self, batchNum):
        batch = self.batchNames[batchNum]
        images,labels = self.extractDataFromBatch(batch)
        imagesR = np.reshape(images, (10000,3,32,32))
        return imagesR,labels
    
    def setUpTrainingData (self, images, labels):
        
        imagesShape = images.shape
        labelsShape = labels.shape
        dimension = len (imagesShape)
        
        labelsReshaped = np.zeros((labelsShape[1], labelsShape[0]))

        for ii in range(labels.shape[0]):
            vector = labels[ii,:]
            labelsReshaped[:,ii] = vector
       
        if dimension == 4:
            # X is shape (10000, 3072)
            N = images.shape[0]
            images = images.reshape(N, 3, 32, 32)          # (10000, 3, 32, 32)
            images = images.transpose(2, 3, 1, 0) # (32, 32, 3, 10000)
        else:
            if dimension == 3: 
                images = ms.reShape3DMatrix(images)
            else:
                images = ms.reShape2DMatrix(images)
        
        return images, labelsReshaped
 
    def getTrainingBatchByNumber(self, batchNumber):

        images , labels = self.getBatchData(self.batchNames[batchNumber])
        trainingImages , trainingLabels = self.setUpTrainingData (images, labels)

        return trainingImages , trainingLabels

    
   
    
    def getTestBatch(self):
        images , labels = self.getBatchData('test_batch')
         # images has shape (10000,32,32,3)
        return images, labels

    def getTrainingBatches(self, numBatches, greyScale):
        
        batchList = []
        for ii in range(numBatches):
            batchList.append(ii)
        trainingImages, trainingLabels=  self.GetBatchDataFromList(batchList, greyScale)
        return trainingImages, trainingLabels

    def getBatchDataFromList(self,batchList):
        
        trainingImages , trainingLabels = self.getBatchData(self.batchNames[batchList[0]])
        
        if len(batchList) > 1:
        
            for ii in range(len(batchList) - 1 ):
        
                images , labels = self.getBatchData(self.batchNames[batchList[ii+1]])
                tImages = np.concatenate((trainingImages,images), axis = 0)
                tLabels = np.concatenate((trainingLabels,labels), axis = 0)
                
                trainingImages =  tImages
                trainingLabels = tLabels   

        trainingImages, trainingLabels = self.setUpTrainingData(trainingImages, trainingLabels)
        
        return trainingImages, trainingLabels
    
    def getBatchDataFromListGPT(self,batchList):
        
        batchNo = batchList[0]
        trainingImages , trainingLabels = self.getBatchDataForGPT([batchNo])
        
        if len(batchList) > 1:
        
            for ii in range(len(batchList) - 1 ):
                batchNo = batchList[ii+1]
                images , labels = self.getBatchDataForGPT([batchNo])
                tImages = np.concatenate((trainingImages,images), axis = 0)
                tLabels = np.concatenate((trainingLabels,labels), axis = 0)
                
                trainingImages =  tImages
                trainingLabels = tLabels   
        
        return trainingImages, trainingLabels
    
    def getTestBatchForGPT(self):
        
        images,labels = self.extractDataFromBatch('test_batch')
        images =  np.reshape( images,(10000,32,32,3))
        images = np.transpose (images,(0,3,1,2))
        #imagesR = np.reshape(images, (-1,3,32,32))
        images = images.astype(np.float32)/255.0
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        X, y = images[indices], labels[indices]
        y = to_categorical(y)   
        
        return X,y

    def getBatchDataForGPT(self,batchList):
        batch = self.batchNames[batchList[0]]
        images,labels = self.extractDataFromBatch(batch)
        images =  np.reshape( images,(10000,32,32,3))
        images = np.transpose (images,(0,3,1,2))
        #imagesR = np.reshape(images, (-1,3,32,32))
        images = images.astype(np.float32)/255.0
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        X, y = images[indices], labels[indices]
        y = to_categorical(y)
    
        return X, y

    def getBatchDataForCNN(self,batchList):
        batch = self.batchNames[batchList[0]]
        images,labels = self.extractDataFromBatch(batch)
        # images has shape (10000,32,32,3)
        images = np.transpose (images,(1,2,3,0))
        #images = np.reshape(images, (32, 32, 3, 10000))
        images = images.astype(np.float32)/255.0
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        X, y = images[indices], labels[indices]
        y = to_categorical(y)
    
        return X, y
                
    def hello(self):
        print (" cifar says Hello!")
        

    def loadFromKeras(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        assert x_train.shape == (50000, 32, 32, 3)
        assert x_test.shape == (10000, 32, 32, 3)
        assert y_train.shape == (50000, 1)
        assert y_test.shape == (10000, 1)
        
    
    def convertToGreyScale(self,image):
        r, g, b = image[:,:,:,0] , image[:,:,:,1], image[:,:,:,2]
        gImage = 0.989 * r + 0.05870 * g + 0.1140 * b
        # normalize 
        max= np.max(gImage)
        min = np.min (gImage)
        nImage = (gImage - min)/(max - min )
        max= np.max(nImage)
        return nImage
    
# tThe format here is that from function extractDataFromBatch
# kimage must be un-normalized.
        
    def displayOneSample(self, images, labels):
        idx = np.random.randint(images.shape[0])
        
        # figure its "type" from the labels
        iType = labels[idx,:]
        labelValue = np.nonzero(iType)
        zeroArray = labelValue[0]
        self.displayedImageType = zeroArray[0]  
        
        if self.showImage == True:
            shape = images.shape
            if len(shape) == 3:
                plt.imshow(images[idx, :,:])
            if len(shape) == 4:
                plt.imshow(images[idx, :,:,:])   

            plt.title('Label: {}'.format(labels[idx]))
            plt.axis('off')
            plt.show()
            
        return idx
    
    def normalizeByChannel(self, matrix,alpha, beta):
        eps = 0.000001
        H, W,C, N = matrix.shape 
        
        
        ar = np.transpose(matrix,(2,3,0,1))
        mu = ar.mean(axis =1, keepdims = True)
        var = ar.std(axis =1, keepdims = True)
        std = np.sqrt(var + eps)
        ar = (ar - mu)/std
        ar_out = ar * alpha + beta
        out = np.transpose(ar_out, (H,W,C,N))
        
        return mu, std, out

    def normalizeByChannelForGPT(self, images):
        
        images =  np.reshape( images,(10000,32,32,3))
        images = np.transpose (images,(0,3,1,2))
        
        images = images/255.0
        images = (images - 0.5)/0.5

        return images
        
