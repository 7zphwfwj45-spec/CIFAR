import numpy as np
#import filters
import sys
from enum import Enum
sys.path.append("/Volumes/MyDrive/sharedPythonFiles")
        
import MatrixScaling as ms
import convolutionUtils as co
import numpy as np

class imageType(Enum):
    single3D = 1
    single2D = 2
    many2D = 3
    many3D = 4
class gatherParameter(Enum):
    dx = 0
    dw = 1


class createCNN ():
    def  __init__(self, batchParameters, layerList):

        # set up the layers
        self.map = {}
        self.layerWeightsDefined = {}
        counter = 1
        for list in layerList:
            self.map[counter] = convolutionLayer(list)
            self.layerWeightsDefined[counter] = False
            counter += 1  
        self.maxKey = len(self.map)

        if len(batchParameters) == 0:
            self.batchJobSize = 0
            self.nBatches = 0
        else:
            self.batchJobSize = batchParameters[0]
            self.nBatches = batchParameters[1]
            
        self.batchIndex = 0
        self.useAdam = False
        self.learningRate = 0.0
        self.standardD = 0.01
        self.batchNormalization = False
        
    def setInputVectorShape(self, inputVectorsShape):
        self.trainingVectorsShape = inputVectorsShape
        
    def setTrainingVectors(self, trainingVectors, trainingResults):
         
        self.trainingVectorsShape = trainingVectors.shape
        self.trainingVectors = trainingVectors
        
        self.nTrainingVectors = 0

        length = len(self.trainingVectorsShape)
        if length == 3:
            self.nTrainingVectors = self.trainingVectorsShape[2]
        else:
            if length == 4:
                self.nTrainingVectors = self.trainingVectorsShape[3]
            else:
                print (" ABORTED!! trainingVectors shape length is incorrect ",length)
                
        self.trainingResults = trainingResults
         

    def initBatchIndex(self):
        self.batchIndex = 0
        self.runningBatchCount = 0
        
    def setStandardD(self, value):
        self.standardD = value
        
    def setBatchNormalization(self,mode):
         self.batchNormalization = mode
         
    def setLearningList(self,learningList):
        self.learningRate = learningList[0]
        self.L2C = learningList[1]
      
    def setLearningRate(self,learningRate):
        self.learningRate = learningRate
        
    def getPredictionFromTraining(self):
    
        self.results = self.allResults
        self.inputVectors = self.trainingVectors
        return self.run(self.inputVectors)

    def getNextBatchJobFor3D(self):
        
# I have to return the resukts

        if self.batchJobSize > 0:

            if self.batchIndex >= self.nTrainingVectors or self.runningBatchCount >= self.nBatches:
                return 0;
            if  len(self.trainingVectorsShape) == 4:
                batchInputVectors = self.trainingVectors[:, :, :, self.batchIndex: self.batchIndex + self.batchJobSize]
                batchInputResults = self.trainingResults[:, self.batchIndex: self.batchIndex + self.batchJobSize]
                self.batchIndex += self.batchJobSize
                return batchInputVectors, batchInputResults
            else:
                batchInputVectors = self.trainingVectors[:, :, :, 1]
                self.batchIndex += self.batchJobSize
                batchInputResults = self.trainingResults
                return batchInputVectors, batchInputResults

        else:
            return batchInputVectors, batchInputResults
        
        
    def setAdamState(self,state):
        self.useAdam = state
        
    
    # example: two layers with two filters each.
    # first Layer output has shape  (13, 13, 2)
    # shape of poolOutArray is  (5, 5, 2)
    # The flattened poolOutVector length  50
    # The output after all vectors have been processed is  (50, 1000)

    def run(self, images):
        
        for ii in range(self.maxKey):
            layer = self.map[ii+1]
            if self.layerWeightsDefined[ii+1] == False:
                layer.useAdam = self.useAdam
                layer.batchNormalization = self.batchNormalization
                layer.defineFilterWeights(images.shape,self.standardD)
                layer.L2C = self.L2C
                self.layerWeightsDefined[ii+1] = True
                
            images= layer.run(images) 
           
        #print (" max imageIn value after a layer run  ",  np.max(images))
            
        flatV = np.reshape(images,(images.shape[0] *  images.shape[1] * images.shape[2], images.shape[3]) )    
       
        return flatV

    def test(self,testImages):
        
        flatOutput= self.run(testImages)
        return flatOutput

    # The input comes from getCNNDataFromFirstLayer from the ANN
    
    def predictTraining(self, flatVectorSize):
        # process in batches else memory consumption gets out of hamd 
        
        flatVectors= np.zeros((flatVectorSize, self.nTrainingVectors))
        self.initBatchIndex()
        localBatchIndex  = 0
        
        for kk in range(self.nBatches):
            batchIputVectors, batchInputResults= self.getNextBatchJobFor3D()
            #mess = "kk is " + str(kk) + " batchIndex is " + str(self.batchIndex)
            #print (mess)
            flatOut = self.test(batchIputVectors)
            flatVectors[:, localBatchIndex: localBatchIndex + self.batchJobSize] = flatOut
            localBatchIndex += self.batchJobSize
        return flatVectors

    def backPropagate( self,djdz):
        
        keys = self.map.keys()
        djdx = djdz
        lastLayer = True
        for ii in reversed(keys):
            layer = self.map[ii]
            djdx = layer.backPropagateLayer(self.learningRate, lastLayer, djdx)
            lastLayer = False
            
        self.updateAllWeightsAndBias()
        
    def computeL2Derivative(self):
        
        L2Derivative  = 0.0
        # we need djdw to account for the presence of L2Regularization
        if self.L2C > 0.0:
            m = self.imageToRun.shape[3]
            L2Derivative = (self.L2C/m) * self.allFilters               
        return L2Derivative    

            
    def updateAllWeightsAndBias( self):
        
        keys = self.map.keys()
        for ii in range(len(keys)):
            layer = self.map.get(ii+1)
            if layer.useAdam == True:
                self.updateWeightsAndBiasForLayerUsingAdam(layer)
            else:
                self.updateWeightsAndBiasForLayer(layer)
                
    def updateWeightsAndBiasForLayer (self,layer):
        
        LR = self.learningRate
        L2Derivative = self.computeL2Derivative(layer)
        
        layer.weights = layer.weights - LR * ( layer.djdw + L2Derivative)
        layer.bias = layer.bias - LR * layer.djdb
            

    def updateWeightsAndBiasForLayerUsingAdam(self,layer):
        
        LR = self.learningRate
        if layer.L2C > 0.0:
            L2Derivative = self.computeL2Derivative(layer)
        else:
            L2Derivative = 0.0
    
        layer.djdw = layer.djdw + L2Derivative 
        
        layer.firstWM = layer.firstBeta  * layer.firstWM  + ( 1.0 - layer.firstBeta) * layer.djdw
        layer.secondWM = layer.secondBeta  * layer.secondWM  + ( 1.0 - layer.secondBeta) * (layer.djdw * layer.djdw)
        m = layer.firstWM/(1 - layer.firstBeta)
        
        layer.firstBM = layer.firstBeta  * layer.firstBM  + ( 1.0 - layer.firstBeta) * layer.djdb
        layer.secondBM = layer.secondBeta  * layer.second


        
    def getVectorOutputSize(self, inputVectorShape, padding):

        layer = self.map[1]
        layerSize = inputVectorShape[0]
        paddingCounter = 0
        
        for ii in range(self.maxKey):
            layer = self.map[ii+1]
            
            #if layer.padding == True:
                #padding = int((layer.filterSize - 1) // 2  )
               # layerSize  += 2*padding


            layerSize += 2*padding[paddingCounter]
            paddingCounter +=1 
            filterOutputSize = (layerSize - layer.filterSize)/layer.filtersStride + 1
            #filterOutputSize = (layerSize - layer.filterSize)/layer.filtersStride + 1
            if layer.poolingSize > 0:
                layerSize = filterOutputSize/layer.poolingSize 
            else:
                layerSize = filterOutputSize
 
        flatVectorSize = layerSize * layerSize * layer.numFilters
        return int(flatVectorSize)

# [0, filtersSizeL1, filtersCountL1, filtersStrideL1, poolSizeL1]

class convolutionLayer:
    def  __init__(self, layerList):

        self.layerNumber = layerList[0] 
        # its 2D for now
        #self.filters = np.zeros((1,1))
        self.filterSize = layerList[1]
        self.numFilters = layerList[2]
        self.allFilters = np.zeros((1,1,1,1))
        self.filtersStride = layerList[3]
        self.padding = layerList[4]
        self.poolingSize = layerList[5]
        self.maxPool = 0

        self.poolingStride = self.poolingSize
        self.bias = np.zeros((1,1))

        # the key is the input vector nunber
        # the value is a list of dictionaries , one per depth layer

        # The key for each dictionary  is the subMatrix number and the value is the
        # coordinates of the input vector from which the max Value was extracted

        # we will also need:
        # djdz
        # X the input vectors to this layer.
        # zS the evaluation of X * K + B where is the array of filters.
        # aS or "C" as in the video; that is Zs evluated by "relu" 
        # P the result of the pooling layer.
        
        # The result of pooling is a multidimensional array
        # just use any dimensions
        self.poolOut = np.array((1,1,1,1))
        self.filterOut = np.array((1,1,1))
        self.normalizedImages = np.array((1,1,1,1))
        self.adjustedNormalizedImages =np.array((1,1,1,1))
        self.mean = np.zeros((1,1,1), np.float64)
        self.var = np.zeros((1,1,1), np.float64)
        
        
        self.alpha = 1.0
        self.beta = 0.0
        self.batchNormalization = False
        
        self.reLuOut = np.array((1,1,1))
        self.imageToRun = np.array((1,1,1))
        self.printStatus = False
        
        self.firstBeta = 0.9
        self.secondBeta = 0.999
        self.epsilon = 10**(-6)
        self.useAdam =  False
        self.L2C = 0.0
        
        # normalization parms
        
        self.bAlpha = np.zeros((self.numFilters))
        self.bBeta = np.zeros((self.numFilters))
        self.djbAlpha = np.zeros((self.numFilters))
        self.djdBeta = np.zeros((self.numFilters))
        self.bAlpha.fill(1.0) 
        self.bBeta.fill(0.0)
        
        self.bAlphaGr = 0.0
        self.bBetaGr = 0.0
        self.djbAlphaGr = 0.0
        self.djdBetaGR = 0.0            


    def setImageType(self, iType):
        self.inputImageType = iType

    def exitProgram(self, mess):
        sys.exit(mess)
    
    def smartPrint(self, mess, object):
        if self.printStatus == True:
            self.printMaxMinOfObject(mess, object)
        

    # filters are four dimensional withe last dimension being the number of filters
    def defineFilterWeights(self, imageShape, standardD):

        sD = np.sqrt(2.0/(self.filterSize * self.filterSize)) 

        self.allFilters  = np.random.normal(0.0, standardD,(self.filterSize, self.filterSize, imageShape[2], self.numFilters))
        self.bias = np.zeros((self.numFilters, 1))
    
        self.firstWeightMoment = np.zeros(self.allFilters.shape)
        self.secondWeightMoment = np.zeros(self.allFilters.shape)
        self.firstBiasMoment =  np.zeros(self.bias.shape)
        self.secondBiasMoment =  np.zeros(self.bias.shape)
        
        self.alpha = np.zeros((self.numFilters,1))
        self.alpha += 1.0 
        self.beta = np.zeros((self.numFilters,1))
 
    def getFilterDimensions(self, allFilters):
        filterShape = allFilters.shape
        size = filterShape[0]
        nFilters = filterShape[3]
        depth = filterShape[2] 
        return size, depth, nFilters

    # the unput images will have a dimension of three or four

    def getImageParameters(self, image):

        depth = 0
        numVectors = 1
        imageShape = image.shape
        size = imageShape[0]
        
        if len(imageShape) == 4:
            numVectors = imageShape[3]
            depth = imageShape[2]
        else:
            numVectors = imageShape(2)
            depth = 1
        return size, depth, numVectors
  

    def normalizeByChannel(self, matrix):
        
        epsilon = 0.000001
        H,W,C,N = matrix.shape
        # flatten all,but the channels 
        t__matrix = np.transpose(matrix,(2,0,1,3))
        flat_matrix = np.reshape(t__matrix, (C,-1))
                       
        mean = np.mean (flat_matrix, axis = 1 ,keepdims = True)
        var = np.var (flat_matrix, axis = 1 , keepdims = True)
        std = np.sqrt(var + epsilon)
        
        normalizedMatrix = (flat_matrix - mean)/std
        
        # the [:,None] inserts a new dimension od size 1 at the specifiec position. 
        # then broadcasting kicks in so if we have [32,1]*[32,100] the second dimension of
        # the firtst matrix will be stretched or broadcast to match the other size
        # the fresult will be [32,100]
        
        out_flat =  self.bAlpha[:, None]*normalizedMatrix + self.bBeta[:,None]
        adjustedNormalizedMatrix = np.reshape(out_flat,(C,H,W,N))
        adjustedNormalizedMatrix = np.transpose(adjustedNormalizedMatrix,(1,2,0,3))
        
        return normalizedMatrix, adjustedNormalizedMatrix, mean, std
    
# gamma is my Alpah, what is x_hat ?, 

    
    
    def run(self,image):

        self.imageToRun = image
        self.filterOut = self.runThruFilters(image, self.allFilters,self.padding, self.filtersStride)
        
        if self.batchNormalization == True:
            self.batchnorm_forward(self.filterOut, self.bAlpha, self.bBeta, 0.000001)
            self.normalizedImages, self.adjustedNormalizedImages, self.mean, self.std = self.normalizeByChannel(self.filterOut)
            self.reLuOut = self.reLu(self.adjustedNormalizedImages)
        else:
            self.reLuOut = self.reLu(self.filterOut)   
        
        if self.poolingSize > 0:
            self.applyPooling(self.reLuOut, self.poolingSize, self.poolingStride)
        else:
            self.poolOut = self.reLuOut      
        
        return self.poolOut
   
    def reLu(self, matrix):
        return (np.maximum(0,matrix))
    
    def largestArrayValue(self, matrix):
        # the matrix is always 2D
        value = -99999.0
        indices = []
        shape = matrix.shape
        for ii in range(shape[0]):
            for jj in range(shape[1]):
                matValue = matrix[ii,jj]
                if  matValue > value:
                    value = matValue
                    indices = [ii, jj]
                    
        return value, indices
                                     
    #-----------------MaxPool------------------------
    def maxpool_forward(self, x, pool_height, pool_width, stride):

        H, W, C, N = x.shape
        H_out = 1 + (H - pool_height) // stride
        W_out = 1 + (W - pool_width) // stride

        out = np.zeros((H_out, W_out, C, N ))
        
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h1 = i * stride
                        h2 = h1 + pool_height
                        w1 = j * stride
                        w2 = w1 + pool_width
                        # gtev the maximum value in the ranges. 
                        out[i, j,c , n] = np.max(x[h1:h2, w1:w2, c, n])
                        
        # "out" contain the maximum value in each of the pools defined by the ranges h1:h2, w1:w2
        
        cache = (x, pool_height, pool_width, stride, out)
        return out, cache

    def applyPooling(self,reLuOut, poolSize, stride):
        
        out, cache = self.maxpool_forward(reLuOut, poolSize, poolSize, stride)

        imageSize, imageDepth, numImages = self.getImageParameters(reLuOut)
        reLuShape = reLuOut.shape
           
        poolRowLen = int((reLuShape[0] - poolSize)/stride + 1)
        poolColLen = int((reLuShape[1] - poolSize)/stride + 1) 
        depth = reLuShape[2]

        self.poolOut =  np.zeros((poolRowLen, poolColLen, depth, numImages))
       
        for jj in range(numImages):
            counter = 0
            matOut = np.zeros((poolRowLen * poolRowLen, imageDepth))
            for rowM in range(0, imageSize - 1 , stride):
                for colM in range(0, imageSize - 1 , stride):  
                    subMatrix = reLuOut[rowM : rowM + poolSize, colM : colM + poolSize, :,jj]
                    flatMatrix = np.reshape(subMatrix,(poolSize *poolSize, imageDepth))
                    maxValues = np.max(flatMatrix, axis = 0)
                    # the number of max values will equal the depth dimension.
                    matOut[counter,:] = maxValues
                    counter += 1
                    
            self.poolOut[:,:,:,jj] = np.reshape(matOut,(poolRowLen, poolColLen,imageDepth ))


    def reLuGradient( self, matrix):
        matrix[matrix <= 0 ] = 0
        matrix[matrix > 0]= 1
        
        return matrix
    
 # vector is 2D; desired shape (x,y,d, nvectors)
    # form in order rows, columns, depth
    def convertVectorToMatrix(self, vector, targetShape):
        vectorShape = vector.shape
        matrix = np.zeros((targetShape))

        numVectors = vectorShape[1]
        for ii in range(numVectors):
            incr = 0
            for jj in range(targetShape[2]):
                for kk in range(targetShape[0]):
                        for ll in range (targetShape[1]):
                            if len(targetShape) == 4:
                                matrix[kk,ll,jj,ii] = vector[incr, ii]
                            else:
                                matrix[kk,ll,jj] = vector[incr, ii]
                            incr += 1
        return matrix

    def batchnorm_backward(self,dout, cache):
        x_hat, mu, var, gamma, beta, eps, orig_shape = cache
        N, C, H, W = orig_shape
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        m = dout_flat.shape[0]
        dx_hat = dout_flat * gamma
        dvar = np.sum(dx_hat * (x_hat * -0.5) / (var + eps), axis=0)
        dmu = np.sum(-dx_hat / np.sqrt(var + eps), axis=0)
        dx = (dx_hat / np.sqrt(var + eps)) + (dvar * 2 * x_hat / m) + (dmu / m)
        dgamma = np.sum(dout_flat * x_hat, axis=0)
        dbeta = np.sum(dout_flat, axis=0)
        dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return dx, dgamma, dbeta


    def updateWeightsAndBias( self,learningRate, djdw, djdb, L2Derivative):
        
        #print (" In CNetwork.update filter weights=-bias with mean djdw value  ",np.mean(djdw))

        if self.useAdam == True:
            self.firstWeightMomemnt = self.firstBeta  * self.firstWeightMoment  + ( 1.0 - self.firstBeta) * djdw
            self.firstBiasMomemnt = self.firstBeta  * self.firstBiasMoment  + ( 1.0 - self.firstBeta) * djdb
        
        
            self.allFilters = self.allFilters - learningRate * (self.firstWeightMomemnt + L2Derivative)
            self.bias = self.bias - learningRate * self.firstBiasMomemnt 
        else:
            self.allFilters = self.allFilters - learningRate * (djdw + L2Derivative)
            self.bias = self.bias - learningRate * djdb
            
        #print (" In CNetwork.update filter weights-bias with maean filter value  ",np.mean(self.allFilters))
        #print (" In CNetwork.update filter weights=-bias withm ean bias value  ",np.mean(self.bias))
        
    def conv_forward(X, W, b, stride=1, padding=0):
        N, C, H, W_in = X.shape
        F, _, HH, WW = W.shape

        H_out = (H + 2*padding - HH) // stride + 1
        W_out = (W_in + 2*padding - WW) // stride + 1

        X_padded = np.pad(
            X,
            ((0,0),(0,0),(padding,padding),(padding,padding)),
            mode='constant'
        )

        out = np.zeros((N, F, H_out, W_out))

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        hs = i * stride
                        ws = j * stride
                        window = X_padded[n, :, hs:hs+HH, ws:ws+WW]
                        out[n, f, i, j] = np.sum(window * W[f]) + b[f]

        cache = (X, W, b, stride, padding, X_padded)
        return out, cache



     
    def conv_backward(dout, cache):
        """
        dout: (N, F, H_out, W_out)
        cache: (X, W, b, stride, padding, X_padded)
        
        Returns:
        dX: (N, C, H, W)
        dW: (F, C, HH, WW)
        db: (F,)
        """

        X, W, b, stride, padding, X_padded = cache
        N, C, H, W_in = X.shape
        F, _, HH, WW = W.shape
        _, _, H_out, W_out = dout.shape

        dX_padded = np.zeros_like(X_padded)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        # Bias gradient
        db = np.sum(dout, axis=(0, 2, 3))

        # Weight & Input gradients
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        hs = i * stride
                        ws = j * stride

                        window = X_padded[n, :, hs:hs+HH, ws:ws+WW]

                        # Weight gradient
                        dW[f] += dout[n, f, i, j] * window

                        # Input gradient
                        dX_padded[n, :, hs:hs+HH, ws:ws+WW] += dout[n, f, i, j] * W[f]

        # Remove padding
        if padding > 0:
            dX = dX_padded[:, :, padding:-padding, padding:-padding]
        else:
            dX = dX_padded

        return dX, dW, db

    def backPropagateLayer(self, learningRate, isLastLayer, djdz):
        
        # djdw =  dj/dz * dz/df * df/dp * dp/da * da/dz * dz/dw
        self.printStatus =  False
        if isLastLayer == True:
            str = " True "
        else:
            str = " False" 
        
        if isLastLayer == True:
            
            # First dj/df = dj/da * da/dz * dz/df
            
            djdf = djdz
            
            # Second we need dj/dp which is simply the rows of dj/df reshaped into that of P.
            # djdp has shape (5,5,6,100) 
            
            djdp = np.zeros((self.poolOut.shape[0], self.poolOut.shape[1], self.poolOut.shape[2] ))
            djdp = np.reshape(np.transpose(djdf), self.poolOut.shape)
        else:   
            djdp = djdz

        # Third we need dj/da = dj/dp * dp/da 
        # this includes processing max pooling
        
        djda = self.backPropPooling(djdp)
        
       
        dadz = self.reLuGradient ( self.filterOut)
        if self.batchNormalization == True:
            temp, self.djbAlphaGr, self.djbBetaGr =self.my_batchnorm_backward(dadz)
        else:
            temp = dadz
        
#       next we backprop thru the convolutions
#       dout is the output of relu_backward if no batchnorm_backward
#       self.conv_backward(self, dout, X, W, b, stride, padding, X_padded)
        dx, dw, db = self.conv_backward (temp, self.imageToRun, self.allFilters, self.bias, self.padding)

        djdz = djda * dadz 
            
        djdw ,djdb = self.figureDWGradient(djdz, self.imageToRun, self.allFilters)
        
        # skip the first layer
        if self.layerNumber != 0:
            djdx = self.figureDXGradient (djdz, self.imageToRun, self.allFilters) 
        else:
            djdx = 0
            
        return djdx
    
    # djdp has the values to back propagate to reLuOut  (5,5,12,100)
    # reLuOut is (10,10,12,100)
    
    def maxpool_backward(dout, cache):
   
        x, pool_height, pool_width, stride, out = cache
        N, C, H, W = x.shape
        H_out, W_out = dout.shape[2], dout.shape[3]

        dx = np.zeros_like(x)
        
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h1 = i * stride
                        h2 = h1 + pool_height
                        w1 = j * stride
                        w2 = w1 + pool_width

                        # Pooling region
                        window = x[n, c, h1:h2, w1:w2]
                        m = np.max(window)
                        
                        # Mask of the max location(s)
                        mask = (window == m)
                        
                        # Distribute the gradient only to the max location(s)
                        dx[n, c, h1:h2, w1:w2] += dout[n, c, i, j] * mask

        return dx

    
    def backPropPooling(self,djdp):
        

        if self.poolingSize == 0:
           djda = djdp
           return djda
        
        reLuOutSize, depth, numVectors = self.getImageParameters(self.reLuOut)
        djdpShape = djdp.shape
        djda = np.zeros((reLuOutSize, reLuOutSize, depth,numVectors))
        poolSize = self.poolingSize
        poolStride = self.poolingStride
 
        for ii in range(numVectors):
            # flatten djdp retaining the depth layers -> (25,12)

            djdpFlat = np.reshape(djdp[:,:,:,ii],(djdpShape[0] * djdpShape[1], djdpShape[2]))
            counter = 0

            # loop thru the pooling subMatrices of reLuOut subMatrx is (2,2,12)

            for rowM in range(0, reLuOutSize - poolSize + 1 , poolStride):
                for colM in range(0, reLuOutSize - poolSize + 1 , poolStride):  

                    subMatrix = self.reLuOut[rowM : rowM + poolSize, colM : colM + poolSize, :, ii]
                    flatMatrix = np.reshape(subMatrix,(poolSize*poolSize, depth))
                    updatedSubMatrix = np.zeros(flatMatrix.shape)
                    djdpSub = djdpFlat[counter,:]
                    
                    # get the max indices of each flatMatrix and their depth layers in reLuOut  -> 12 
                    indices= np.argmax(flatMatrix, axis = 0)

                    # using the values from djdp to update reLuOut ; djdaSub is (12,),uodatedSubMatrix is (4,12)
                    # djdpSub contains the values for all depth layers of a single entry in djdp
                      
                    # for each index , ii,  generated by "depth" , set updatedSubMatrix[ii, indices[ii]= curr_dout_region
                    updatedSubMatrix[indices,np.arange(depth) ] = djdpSub

                    reShaped = np.reshape(updatedSubMatrix, (poolSize, poolSize, depth))
                    djda[rowM:rowM + poolSize, colM: colM + poolSize,:, ii] = reShaped
                    counter += 1 

        return djda
 
    # let "D" be the filter corresponding to a depth of "D"
    # djdw[i,j,k] = dj/dz[m,n,D] * dz(m,n,D)/dw[i,j,k]
    # so for a particular weight in the filter find all aZ
    # such that it particpates in and sum

    # 1. djdw should have shape (5,5,4, 6) 
    # 2. imageToRun is  (14, 14, 4, 100)
    # 3. djdz is  (10, 10, 6, 100)   

    def runThruFilters(self, images, allFilters, padding, stride):

        filterSize, filterDepth, numFilters = self.getFilterDimensions(allFilters)
        
        if padding == True:
            paddingSize = (filterSize - 1) // 2  
        else:
            paddingSize = 0
            
        paddingSize = padding
            
        paddedImages = np.pad(images,((paddingSize, paddingSize), (paddingSize, paddingSize), (0, 0), (0, 0) ), mode='constant')
        paddedImageShape = paddedImages.shape
        
        imageSize, imageDepth, numImages = self.getImageParameters(images)
        nSubMatrices = (paddedImageShape[0] - filterSize + 1 )
        
        filterOut = np.zeros((nSubMatrices, nSubMatrices, numFilters,numImages))

        image_col = np.zeros((filterSize * filterSize * imageDepth, nSubMatrices *   nSubMatrices))
        flatFilters =np.reshape(self.allFilters, (filterSize * filterSize * imageDepth, numFilters))
        
        for jj in range(numImages):
            imageColCounter = 0     
            for rowM in range(0, paddedImageShape[0]- filterSize + 1 , stride):
                for colM in range(0, paddedImageShape[1] - filterSize + 1, stride): 
                    subMatrix = paddedImages[rowM : rowM + filterSize, colM : colM + filterSize, :, jj]
                    flatSubMatrix = np.reshape(subMatrix,(filterSize *  filterSize * imageDepth))
                    image_col[:,imageColCounter] = flatSubMatrix
                    imageColCounter += 1  

            #  for first layer: flatFilters weights are  (75,32) is (5X5X3)and (32 the number of filters)
            #  image_col is (75,1024) 75 is the image sub matrix ; 784 is the number of sub matrices 
            #   values shape is (32,784)
            values = np.dot(np.transpose(flatFilters), image_col) + self.bias

            values = np.reshape(np.transpose(values),(nSubMatrices, nSubMatrices, numFilters ))
            filterOut[:,:,:,jj] = values
             
        return filterOut

    def figureDXGradient(self, djdz, images, allFilters):
             
        # djdz is (8,8,64,32)
        # image shape is (8,8,64,32)
        # weights (3,3,16,16 )- (3,3,16,32)
        
        # padded shape is (20,20,6,100)
        # he has djdz depth = input vector depth = weight depth =  number of filter  = 64
        #  I have input vector depth = weight depth = 4 this is necessary
        #  djdz depth = number of filters = 12 
        
        djdzShape = djdz.shape
        imageShape = images.shape
        filterSize, filterDepth, numFilters = self.getFilterDimensions(self.allFilters)
        
        #padding =  int((filterSize - 1) // 2 )
        padding = self.padding
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
    
    def figureDWGradient(self, djdz, images, allFilters):
        # djdz is (8,8,64,32)
        # image shape is (16,16,12,63) 
        # filters (5,5,64,64)
    
        filterShape = allFilters.shape
        filterSize = filterShape[0]
        filterDepth = filterShape[2]
        numFilters = filterShape[3]

        imageShape = images.shape
        imageSize = imageShape[0]
        imageDepth = imageShape[2]
        numImages = imageShape[3]

        djdzShape = djdz.shape
        djdzSize = djdzShape[0]
        djdzDepth = djdzShape[2]
        djdzNum = djdzShape[3]
        
        # padding as necessary so that the dot product inputs have compatible shapes
         
        #padding =  int((filterSize - 1) // 2 )   
        padding = self.padding
        paddedImages = np.pad(images, ((padding, padding), (padding, padding), (0, 0), (0, 0) ), mode='constant')
        paddedShape = paddedImages.shape
        bounds = paddedShape[0] - filterSize + 1

        stride = 1 
        djdw = np.zeros((filterSize, filterSize, filterDepth, numFilters)) 
        dbias = np.zeros((self.bias.shape))
       
        for nn in range(numImages):
            
            counter = 0
            adjdz = djdz[:,:,:,nn]
            # (256,16)
            flat_djdz = np.reshape(adjdz,(djdzSize * djdzSize, djdzDepth))

            # (108,256),(3*3*12.16),allocate space for all sub matrices for this image e.g dz/dw
            allSubMatrices= np.zeros((filterSize * filterSize * filterDepth, imageSize * imageSize))
            
            for rowM in range(0, bounds, stride):
                for colM in range(0, bounds, stride):
                    subMatrix =  paddedImages[rowM:rowM + filterSize, colM:colM + filterSize,:, nn]
                    flat_subMatrix = np.reshape(subMatrix,(filterSize * filterSize * filterDepth))
                    allSubMatrices[:,counter] = flat_subMatrix
                    counter += 1 
            
            # (108,256)(256,16) -> (106,16) dj/dw = dj/dz. 
            dot = np.dot(allSubMatrices,flat_djdz)
            flat_dot = np.reshape(dot,filterShape)
            djdw += flat_dot
            
            # delta is (12,)  flat_djdz is (256,12) axis = 0 sums over columns
            delta = np.sum(flat_djdz, axis = 0)
            deltaR = np.reshape(delta,(delta.shape[0], 1))
            dbias += deltaR
            
        return djdw, dbias
