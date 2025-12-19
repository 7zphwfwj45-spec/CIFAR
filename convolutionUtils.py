import numpy as np
#import filters
import sys
from enum import Enum
sys.path.append("/Volumes/MyDrive/sharedPythonFiles")
        
import MatrixScaling as ms


def convolveWithPadRotate(images, filters, paddingSize, stride):

    paddedImages = np.pad(images,((paddingSize, paddingSize), (paddingSize, paddingSize), (0, 0), (0, 0) ), mode='constant')
    numFilters = filters.shape[3]
    
    rotatedFilters =  np.zeros((filters.shape))
    for ii in range(numFilters):
         rotatedFilters[:,:,:,ii] = ms.singleRotate3D(filters[:,:,:,ii])
    convolveOut = convolve(paddedImages, rotatedFilters, stride)
    
    return convolveOut  
    

def convolve(images, filters, stride):
    
    filtersShape = filters.shape
    imageShape = images.shape 
    filterSize = filtersShape[0]
    filterDepth = filtersShape[2]
    
    imageShape = images.shape 
    imageSize = imageShape[0]   
    numImages = imageShape[3]
    
    fOutSize = int((imageSize - filterSize)/stride + 1)
    convolveOut = np.zeros((fOutSize,fOutSize, filterDepth,numImages))
    
    for ii in range(numImages):
        convolveOut[:,:,:,ii] = convolveMulti(images[:,:,:,ii], filters, stride)

    return convolveOut
    
def convolveMulti(image,filters, stride):
    
    filtersShape = filters.shape
    imageShape = image.shape 
    filterSize = filtersShape[0]
    filterDepth = filtersShape[2]
    numFilters = filtersShape[2]
    
    imageShape = image.shape 
    imageSize = imageShape[0]   
    
    fOutSize = int((imageSize - filterSize)/stride + 1)
    multiOut = np.array((fOutSize,fOutSize, filterDepth))
    
    for ii in range(numFilters):
        multiOut[:,:,ii] = convolveSingle(image,filters[:,:,:,ii],stride)

    return multiOut

def convolveSingle(image, filter, stride):
    
        filterShape = filter.shape
        filterSize = filterShape[0]
        filterDepth = filterShape[2]
        imageShape = image.shape 
        imageSize = imageShape[0]  
        
        imageDepth = imageShape[2]
        outSize = int((imageSize - filterSize)/stride + 1)
        filterOut = np.zeros((outSize, outSize))

        flatFilter =np.reshape(filter, (filterSize * filterSize * filterDepth))
          
        for rowM in range(0, imageSize - filterSize + 1 , stride):
            for colM in range(0, imageSize  - filterSize + 1, stride): 
                subMatrix = image[rowM : rowM + filterSize, colM : colM + filterSize, :]
                flatSubMatrix = np.reshape(subMatrix,(filterSize *  filterSize * imageDepth))
                # (1536,) (3072)
                dot = np.dot(flatSubMatrix,flatFilter) 
                filterOut[rowM, colM] = dot
        return filterOut

def figureDXGradient(self, djdz, images, allFilters):
             
    # djdz is (16,16,16,64) - (16,16,32,64)
    # image shape is (16,16,16,64) -(16,16,16,64)
    # weights (3,3,16,16 )- (3,3,16,32)
        
    # padded shape is (20,20,6,100)
    # he has djdz depth = input vector depth = weight depth =  number of filter  = 64
    #  I have input vector depth = weight depth = 4 this is necessary
    #  djdz depth = number of filters = 12 
        
    djdzShape = djdz.shape
    imageShape = images.shape
    filterSize, filterDepth, numFilters = self.getFilterDimensions(self.allFilters)
        
    padding =  int((filterSize - 1) // 2 )
    paddedImages = np.pad(images, ((padding, padding), (padding, padding), (0, 0), (0, 0) ), mode='constant')
    paddedShape = paddedImages.shape
    bounds = paddedShape[0] - filterSize + 1
    stride = 1 
        
    # (16,16,16,64
    djdx = np.zeros((imageShape))
        
        # flatten the weights to (144,16)
    flatWeights = np.reshape(self.allFilters,(filterSize*filterSize*imageShape[2],numFilters))
        
    for nn in range(imageShape[3]):

        counter = 0
        temp_dx = np.zeros((paddedShape[0],paddedShape[1],imageShape[2]))
        #( 16,16,12)
        adjdz  = djdz[ :, :, :,nn]
        # (256,16) why numFilters?? djdz depth = numFilters and imagesize = djdz size
        reshaped_adjdz = np.reshape(adjdz,(imageShape[0] * imageShape[0], numFilters))
            
        # this dot product is in effect dj/dz dz/dx; dj/dx is simply the weights
        # (256,16)(144,16) -> (256,16)(16,144) -> (256,144)
        temp = np.dot(reshaped_adjdz, np.transpose(flatWeights)) 
                                                                                                                                                                                                               
        for rowM in range(0, bounds , stride):
            for colM in range(0, bounds, stride):
                # take each row and (144,)
                aTemp = temp[counter,:]
                reshaped_aTemp = np.reshape(aTemp, (filterSize,filterSize,imageShape[2]))
                # (3,3,16)
                temp_dx[rowM:rowM +filterSize, colM:colM + filterSize,:] +=  reshaped_aTemp
                counter +=1 
            # (16,16,16)    
        djdx[:,:,:,nn] = temp_dx[padding:-padding, padding:-padding, :]

    return djdx
    