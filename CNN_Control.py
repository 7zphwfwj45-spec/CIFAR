import sys
sys.path.append("/Volumes/MyDrive/sharedPythonFiles")
import numpy as np
from numpy import linalg as LA
import AdamOptimizer as ao
from datetime import datetime
import pickle
import math as mth

class Control:
    def __init__(self, batchSize, lr , epochs, dropOutRate, weightDecay, gapInUse):
        
        self.X = 0
        self.Y = 0
        self.batchSize = batchSize
        self.lr = lr
        self.epochs = epochs
        
        self.dropOutRate = dropOutRate
        self.dropOutByLayer = [False, False, False]
        self.denseBatchNormalization = [False, False, False]
        self.Y_batch_results = 0.0
        self.params = {}
        self.weight_decay= weightDecay
        self.loss = []
        self.runningCost = []
        self.augmentation = False
        self.gapInUse = gapInUse


        print (" Created Convolution Class")
        
        # go to this configuration
        #   32 → 32 → pool
        #   64 → 64 → pool
        
        self.conv_spec_list = [
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

        #self.dense_channels = [256, 10]  
        # for GAP there is only the "10" layer.
        self.dense_channels = [10]
        
    def setTrainingData(self, X,Y):
        self.X = X
        self.Y = Y
        self.Y_batch_results = np.zeros(Y.shape)

    def setAugmentation(self, augmentation):
        self.augmentation = augmentation
            
    def setLearningRate(self, lr):
        self.lr = lr
        
    def getCost(self):
        return self.runningCost
    
    def setDenseBatchNorm(self, status):
        self.denseBatchNormalization = status
        
    def setDropout(self, rate, layers):
        self.dropOutByLayer = layers
        self.dropOutRate = rate
        
    def setEpoch(self,currentEpoch):
        self.epoch = currentEpoch
        
    def cosineDecay(self, maxLr, minLr, totalCount):
        self.maxLr = maxLr
        self.minLr = minLr
        self.totalEpochs = totalCount

        
# delete a instantioted class instance.
    def __del__(self):
        print ("f{self.name} destroyed")     


    def im2col(self, x, HH, WW, stride, pad):
        N, C, H, W = x.shape
        H_out = (H + 2*pad - HH) // stride + 1
        W_out = (W + 2*pad - WW) // stride + 1

        x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))

        cols = np.zeros((N, C, HH, WW, H_out, W_out))

        for i in range(HH):
            i_max = i + stride * H_out
            for j in range(WW):
                j_max = j + stride * W_out
                cols[:, :, i, j, :, :] = x_padded[:, :, i:i_max:stride, j:j_max:stride]

        cols = cols.transpose(0,4,5,1,2,3).reshape(N*H_out*W_out, -1)
        return cols


    def conv_forward(self, x, w, b, stride=1, pad=1):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape

        H_out = (H + 2*pad - HH) // stride + 1
        W_out = (W + 2*pad - WW) // stride + 1

        x_cols = self.im2col(x, HH, WW, stride, pad)
        w_cols = w.reshape(F, -1)

        out = x_cols @ w_cols.T + b
        out = out.reshape(N, H_out, W_out, F).transpose(0, 3, 1, 2)

        cache = (x, w, b, stride, pad, x_cols)
        return out, cache

 
    def dense_batchnorm_forward(self, x, gamma, beta, running_mean, running_var,inTraining,
                        momentum=0.9, eps=1e-5):
        
        if inTraining == True:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            std = np.sqrt(var + eps)

            x_hat = (x - mu) / std

            running_mean[:] = ( momentum * running_mean + (1 - momentum) * mu)
            running_var[:] = (momentum * running_var + (1 - momentum) * var)
            #x, x_hat, mu, var, std, gamma = cache
            cache = (x, x_hat, mu, var, std, gamma)

            #cache = ("dense", x_hat, std, gamma)

        else:
            x_hat = ((x - running_mean)/ np.sqrt(running_var + eps))
            cache = None

        out = gamma * x_hat + beta

        return out, cache

    def conv_batchnorm_forward(self, x, gamma, beta,
                            running_mean, running_var,
                            inTraining, momentum=0.9, eps=1e-5):

        N, C, H, W = x.shape
        x_flat = x.transpose(1,0,2,3).reshape(C, -1)

        gamma = gamma.reshape(-1,1)
        beta  = beta.reshape(-1,1)

        if inTraining:
            # 1D for running stats
            mu_flat  = x_flat.mean(axis=1)        # shape (C,)
            var_flat = x_flat.var(axis=1)         # shape (C,)

            running_mean[:] = momentum * running_mean + (1 - momentum) * mu_flat
            running_var[:]  = momentum * running_var  + (1 - momentum) * var_flat

            # 2D for normalization
            mu  = mu_flat[:, None]                # (C,1)
            var = var_flat[:, None]
        else:
            mu  = running_mean[:, None]
            var = running_var[:, None]

        std = np.sqrt(var + eps)
        x_hat = (x_flat - mu) / std

        out_flat = gamma * x_hat + beta
        out = out_flat.reshape(C, N, H, W).transpose(1,0,2,3)

        cache = (x_hat, std, gamma)
        return out, cache
    

    def conv_batchnorm_backward(self, dout, cache):
        """
        dout : (N, C, H, W)
        """

        x_hat, std, gamma = cache
        N, C, H, W = dout.shape

        # normalize by channel
        dout_flat = dout.transpose(1, 0, 2, 3).reshape(C, -1)

        dbeta = dout_flat.sum(axis=1)
        dgamma = (dout_flat * x_hat).sum(axis=1)
        dxhat = dout_flat * gamma#[:, None]
        
        M = dxhat.shape[1]

        dx = (
            (dxhat / std)
            - (dxhat.sum(axis=1, keepdims=True) / M) / std
            - (x_hat * (dxhat * x_hat).sum(axis=1, keepdims=True) / M) / std
        )
        dx = dx.reshape(C, N, H, W).transpose(1, 0, 2, 3)
        
        """
        dx2, dgamma2, dbeta2 = self.conv_other_batchnorm_backward(dout, cache)
        dxDiff = np.sum(np.subtract(dx, dx2))
        gammaDiff = np.sum(np.subtract(dgamma,dgamma2))
        betaDiff = np.sum(np.subtract(dbeta, dbeta2))
        print ( "diifs are ", dxDiff,gammaDiff,betaDiff)
        """
        return dx, dgamma, dbeta

    def maxpool_forward(self, x, pool_size=2, stride=2):
        N, C, H, W = x.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1

        x_reshaped = x.reshape(
            N, C,
            H_out, pool_size,
            W_out, pool_size
        )

        out = x_reshaped.max(axis=(3,5))

        cache = (x, pool_size, stride, x_reshaped)
        return out, cache

    def dense_batchnorm_backward(self, dout, cache, eps = 1e-5):
        """
        Dense-layer BatchNorm backward pass.

        Inputs:
        - dout: (N, D)
        - cache from forward

        Returns:
        - dx: (N, D)
        - dgamma: (D,)
        - dbeta: (D,)
        """
        x, x_hat, mu, var, std, gamma = cache
        N, D = dout.shape

        # Gradients for scale and shift
        dbeta = np.sum(dout, axis=0)              # (D,)
        dgamma = np.sum(dout * x_hat, axis=0)     # (D,)

        dxhat = dout * gamma                      # (N, D)

        # Backprop through normalization
        dvar = np.sum(dxhat * (x - mu) * -0.5 * (var + eps)**(-1.5), axis=0)
        dmu = (
            np.sum(dxhat * -1 / std, axis=0) +
            dvar * np.mean(-2 * (x - mu), axis=0)
        )

        dx = (
            dxhat / std +
            dvar * 2 * (x - mu) / N +
            dmu / N
        )

        return dx, dgamma, dbeta


    def relu_forward(self, x):
        out = np.maximum(0, x)
        cache = x
        return out, cache

    def relu_backward(self, dout, cache):
        x = cache
        dx = dout * (x > 0)
        return dx
    

    def maxpool_backward(self, dout, cache):
        x, pool_size, stride, x_reshaped = cache
        N, C, H, W = x.shape
        H_out, W_out = dout.shape[2], dout.shape[3]

        # Create mask of max locations
        max_mask = (x_reshaped == x_reshaped.max(axis=(3,5), keepdims=True))

        # Expand dout to match pool window shape
        dout_expanded = dout[:, :, :, None, :, None]

        # Route gradients
        dx_reshaped = max_mask * dout_expanded

        # Reshape back to input shape
        dx = dx_reshaped.reshape(x.shape)

        return dx
    
    def col2im(self, cols, x_shape, HH, WW, stride, pad):
        N, C, H, W = x_shape
        H_out = (H + 2*pad - HH) // stride + 1
        W_out = (W + 2*pad - WW) // stride + 1

        cols = cols.reshape(N, H_out, W_out, C, HH, WW).transpose(0,3,4,5,1,2)
        x_padded = np.zeros((N, C, H + 2*pad, W + 2*pad))

        for i in range(HH):
            i_max = i + stride * H_out
            for j in range(WW):
                j_max = j + stride * W_out
                x_padded[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]

        return x_padded[:, :, pad:pad+H, pad:pad+W]


    def conv_backward(self, dout, cache):
        x, w, b, stride, pad, x_cols = cache
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape

        # db
        db = np.sum(dout, axis=(0, 2, 3))

        # reshape dout
        dout_cols = dout.transpose(0,2,3,1).reshape(-1, F)

        # dw
        dw = dout_cols.T @ x_cols
        dw = dw.reshape(w.shape)

        # dx
        w_cols = w.reshape(F, -1)
        dx_cols = dout_cols @ w_cols
        dx = self.col2im(dx_cols, x.shape, HH, WW, stride, pad)

        return dx, dw, db

    def computeMultipleCost(self):
        m = self.X.shape[0]
        # known as the cross entropy loss
        cost = (-1.0/m)* np.sum(self.Y * np.log(self.Y_batch_results + 1e-12))
        return cost
    
    # logits is my "z"
    def softmax_loss(self, logits, y_true_onehot):
        """
        logits: (N, C) these are my results
        y_true_onehot: (N, C)
        """
        # this figures the probabilities of my results
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        # convert one-hot to a simple integer index
        y_idx = np.argmax(y_true_onehot, axis=1)
        N = logits.shape[0]

        # cross entropy loss
        loss = -np.mean(np.log(probs[np.arange(N), y_idx] + 1e-12))

        # gradient wrt logits
        dlogits = probs.copy()
        
        # this picks the dlogits element from dlogits probabilities using the y_idx index and subtracting one,
        # it picks from the each row of logits the element correspondoing to the actual result.
        dlogits[np.arange(N), y_idx] -= 1
       
        dlogits /= N
        
        # dlogits is computing by djdz on the final layer. THis the equivalent of "Result = predicted - actual"

        return loss, dlogits, probs

    
    def dropout_forward(self, x, p=0.5):
        
        mask = (np.random.rand(*x.shape) > p).astype(x.dtype)/(1-p)
        out = x * mask
        cache = (mask, p, True)
        return out, cache


    def dropout_backward(self, dout, cache):
        mask, p, training = cache
        if not training or p == 0:
            return dout
        else:
            return dout * mask
        
    # x is (4,128) W is (128,10) b is (10,)
    def dense_layer_forward(self,x, W, b):
    
        scores = x @ W + b  
        cache = (x, W, b)
        return scores, cache

    # dout is the output from relu_backward
    def dense_backward(self,dout, cache):
        x, W, b = cache
        dW = x.T @ dout
        db =  np.sum(dout, axis=0)
        dx = dout @ W.T
        return dx, dW, db
   
    def cnn_backward(self,dscores, caches):
       
        grads = {}
        dense_caches = caches['dense_caches']
        conv_caches = caches['conv_caches']
        N, C, H, W = caches['flatten_shape']

        conv_spec_list = caches['conv_spec_list']
        

        # Dense backward
        cur = dscores
        for idx in reversed(range(len(dense_caches))):
            cache_affine, bn_cache, cache_relu, dropout_cache = dense_caches[idx]
        
            # dropout precceds relu
            if dropout_cache is not None:
                cur = self.dropout_backward(cur, dropout_cache)
                
            if cache_relu is not None:
                cur = self.relu_backward(cur, cache_relu)
                
            if self.denseBatchNormalization[idx] == True:
                dz, dgamma, dbeta = self.dense_batchnorm_backward(cur, bn_cache)
                grads[f"gamma_dense_{idx+1}"] = dgamma
                grads[f"beta_dense_{idx+1}"] = dbeta
            else:
                dz = cur
                
            dx, dW, db = self.dense_backward(dz, cache_affine)
            grads[f"W_dense_{idx+1}"] = dW
            grads[f"b_dense_{idx+1}"] = db
            cur = dx

        if self.gapInUse == True:
            cur = self.gapBackward(cur, caches['flatten_shape'])
        else:     
            cur = cur.reshape(N, C, H, W)
        
        # Conv backward
        dx_conv_input, conv_grads = self.conv_stack_backward(cur, conv_caches, conv_spec_list)
        # make sure this called for conv 
        grads.update(conv_grads)

        return grads
    
    def init_params(self,conv_spec_list, dense_channels):
        """
        Initialize params for conv stack + dense stack.

        conv_spec_list: list of dicts with keys:
            - 'filter_size' (F)
            - 'stride'
            - 'pad'
            - 'in_ch' (C_in)
            - 'out_ch' (number of filters)
            - 'use_bn' (bool)
            - 'pool_param' (None or pool param dict)
        dense_channels: list of ints, e.g. [512, 10]
        input_shape: (C, H, W) for a single image (channel-first)

        Returns params dict with keys W_conv_i, b_conv_i, (gamma_conv_i, beta_conv_i),
        and W_dense_j, b_dense_j initialized where possible. Note: W_dense_1 is
        left uninitialized because its input dimension depends on conv output spatial
        size. Use `init_dense_from_sample` to initialize dense weights after running
        a sample forward pass through conv stack.
        """

        params = {}
        # conv params
        for i, spec in enumerate(conv_spec_list, start=1):
            F = spec['filter_size']
            C_in = spec['in_ch']
            C_out = spec['out_ch']
            # He init could be used, but a small normal init is stable for demos
            standardD = 0.015
            params[f"W_conv_{i}"] = np.random.normal(0.0, standardD, (C_out, C_in, F, F))
            #params[f"b_conv_{i}"] = np.zeros((C_out,1))
            params[f"b_conv_{i}"] = np.zeros((C_out))
            if spec.get('use_bn', False):
                #params[f"gamma_conv_{i}"] = np.ones((C_out,1))
                #params[f"beta_conv_{i}"] = np.zeros((C_out,1))
                #params[f"running_mean_{i}"] =  np.zeros((C_out,1))
                #params[f"running_var_{i}"] =  np.ones((C_out,1))
                params[f"gamma_conv_{i}"] = np.ones((C_out,))
                params[f"beta_conv_{i}"] = np.zeros((C_out,))
                params[f"running_mean_{i}"] =  np.zeros((C_out,))
                params[f"running_var_{i}"] =  np.ones((C_out,))
                

        # Dense params: only shapes for layers 2.. are inferable now; W_dense_1 will be
        # created later once conv output flattened size is known
        for j in range(1, len(dense_channels)):
            # He initialization
            standardD = np.sqrt(2.0/( dense_channels[j]))
            params[f"W_dense_{j+1}"] = np.random.normal(0.0, standardD,(dense_channels[j-1], dense_channels[j]))
            params[f"b_dense_{j+1}"] = np.zeros(dense_channels[j])

        for j in range(len(dense_channels)): 
            params[f"gamma_dense_{j+1}"] = np.ones((dense_channels[j]))
            params[f"beta_dense_{j+1}"] = np.zeros((dense_channels[j]))
            params[f"running_mean_dense_{j+1}"] = np.zeros((dense_channels[j]))
            params[f"running_var_dense_{j+1}"] = np.zeros((dense_channels[j]))
            
        return params
    

    def init_dense_from_sample(self,params, conv_stack_forward_fn, x_sample, conv_spec_list, dense_channels, wscale):
        
        """
        Run conv stack forward on x_sample to find flatten size and initialize
        W_dense_1 and b_dense_1 (if missing).

        conv_forward_fn: a callable that runs ONLY the conv stack and returns the
            conv output (N, C, H, W) and optionally caches if needed. We'll call it
            with the same conv_spec_list and params to determine shape.
        x_sample: sample input of shape (N, C, H, W)
        
        """
       
        inTraining = False
        conv_out, _ = conv_stack_forward_fn(x_sample, params, conv_spec_list, inTraining)
        
        if self.gapInUse == True:
            gapOut = self.gapForward(conv_out)
            flattened = gapOut.shape[1]
        else:
            N, C_last, H_last, W_last = conv_out.shape
            flattened = C_last * H_last * W_last

        if "W_dense_1" not in params:
            standardD = np.sqrt(2.0/( dense_channels[0]))        
            params[f"W_dense_1"] = np.random.normal(0.0, standardD,(flattened, dense_channels[0]))
            params["b_dense_1"] = np.zeros(dense_channels[0])

        return params

    def conv_stack_forward(self,x, params, conv_spec_list, inTraining):
        """
        Runs the conv -> (bn) -> relu -> (pool) for every conv layer in conv_spec_list.
        Returns final output and caches for the conv stack only.
        """
        cur = x
        conv_caches = []
        for i, spec in enumerate(conv_spec_list, start=1):
            W = params[f"W_conv_{i}"]
            b = params[f"b_conv_{i}"]
            #conv_param = {'stride': spec.get('stride', 1), 'pad': spec.get('pad', 0)}
            stride = spec.get('stride', 1)
            pad = spec.get('pad', 0)
            out_conv, cache_conv = self.conv_forward(cur, W, b, stride, pad)
            
            status = spec.get('use_bn')
            
            if spec.get('use_bn', False):
                
                out_bn, cache_bn= self.conv_batchnorm_forward(out_conv, 
                  params[f"gamma_conv_{i}"],
                  params[f"beta_conv_{i}"] ,                                                    
                  params[f"running_mean_{i}"],
                  params[f"running_var_{i}"],
                  inTraining)
                    
            else:
                out_bn, cache_bn = out_conv, None

            out_relu, cache_relu = self.relu_forward(out_bn)

            if spec.get('pool_param', None) is not None:
                pool_param = spec['pool_param']
                pool_size = pool_param.get('pool_height')
                if pool_size != 0:
                    stride  = pool_param.get('stride')
                    out_pool, cache_pool = self.maxpool_forward(out_relu, pool_size, stride)
                else:
                    out_pool, cache_pool = out_relu, None
            else:
                out_pool, cache_pool = out_relu, None

            conv_caches.append((cache_conv, cache_bn, cache_relu, cache_pool))
            cur = out_pool

        return cur, conv_caches

   
    def cnn_forward(self,x, params, conv_spec_list, dense_channels, inTraining):
            """
            Full forward: conv stack (vectorized over batch) -> flatten -> dense stack
            Returns scores and a cache dict used by cnn_backward.
            """
            caches = {}

            # conv stack forward
            conv_out, conv_caches = self.conv_stack_forward(x, params, conv_spec_list, inTraining)
            caches['conv_caches'] = conv_caches
            N, C, H, W = conv_out.shape
            if self.gapInUse == False:
                flat = conv_out.reshape(N, C * H * W)
            else:
                flat = self.gapForward(conv_out)
                
            caches['flatten_shape'] = (N, C, H, W)   
            # dense forward
            scores, caches = self.cnn_dense_forward(params, conv_spec_list, flat, self.dense_channels, caches, inTraining)

            return scores, caches

    def cnn_dense_forward (self, params, conv_spec_list, flat, dense_channels, caches, inTraining):
        prev = flat
        num_dense = len(dense_channels)
        dense_caches = []
        
        for i, dch in enumerate(dense_channels, start=1):
                
            Wd = params[f"W_dense_{i}"]
            bd = params[f"b_dense_{i}"]
            gamma = params[f"gamma_dense_{i}"]
            beta = params[f"beta_dense_{i}"]
            runningMean = params[f"running_mean_dense_{i}"]
            runningVar = params[f"running_var_dense_{i}"]
            out_affine, cache_affine = self.dense_layer_forward(prev, Wd, bd)
                
            if i < num_dense:        
                if self.denseBatchNormalization[i-1]: 
                    out_bn, bn_cache = self.dense_batchnorm_forward(out_affine, gamma, beta, runningMean, runningVar, inTraining)
                    out_relu, cache_relu = self.relu_forward(out_bn)
                else:   
                    out_relu, cache_relu = self.relu_forward(out_affine)         
                    bn_cache = None 
                
                if self.dropOutByLayer[i-1] == True : 
                    dropOut, dropout_cache = self.dropout_forward(out_relu, self.dropOutRate)
                    prev = dropOut
                    dense_caches.append((cache_affine, bn_cache, cache_relu, dropout_cache))
                else:
                    prev = out_relu
                    dense_caches.append((cache_affine, bn_cache, cache_relu, None))
            else:
                dense_caches.append((cache_affine, None, None, None))
                prev = out_affine
                
        scores = prev
        caches['dense_caches'] = dense_caches
        caches['conv_spec_list'] = conv_spec_list
            
        return scores, caches


    def conv_stack_backward(self,dout, conv_caches, conv_spec_list):
        """
        Backprop through conv stack. Returns gradient w.r.t input to conv stack and
        gradients dict for conv parameters.
        """
        grads = {}
        cur_grad = dout
        # go in reverse
        for i in range(len(conv_caches)-1, -1, -1):
            cache_conv, cache_bn, cache_relu, cache_pool = conv_caches[i]
            spec = conv_spec_list[i]

            # pool backward
            if cache_pool is not None:
                cur_grad = self.maxpool_backward(cur_grad, cache_pool)

            # relu backward
            cur_grad = self.relu_backward(cur_grad, cache_relu)

            # bn backward
            if spec.get('use_bn', False):
                cur_grad, dgamma, dbeta = self.conv_batchnorm_backward(cur_grad, cache_bn)
                grads[f"gamma_conv_{i+1}"] = dgamma
                grads[f"beta_conv_{i+1}"] = dbeta

            # conv backward
            dx, dW, db = self.conv_backward(cur_grad, cache_conv)
            grads[f"W_conv_{i+1}"] = dW
            grads[f"b_conv_{i+1}"] = db

            cur_grad = dx

        return cur_grad, grads
    
    def train_cnn(self, X, y, params, batch_size, epochs, inTraining):
        
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        self.Y_batch_results = np.zeros(y.shape)

        for epoch in range(epochs):
            
            if epoch > 75:
                self.lr = 0.0003
                
            for b in range(num_batches):
                start = b * batch_size
                end = start + batch_size
                X_batch, y_batch = X[start:end], y[start:end]
                # augment as needed
                if self.augmentation == True:
                    X_batch = self.augment_cifar10(X_batch)

                # scores is the ouput from the final layer. 
                scores, cache = self.cnn_forward(X_batch, params, self.conv_spec_list, self.dense_channels, inTraining)
                # dscores is the gradien wrt to the probabilities
                loss, dscores, probs =  self.softmax_loss(scores, y_batch)
        
                # store the results of this batch
                self.Y_batch_results[start:end, :] = probs
                
                if inTraining == True:
                    grads = self.cnn_backward(dscores, cache)
                    params = self.optimizer.step(self.params, grads)
                    
            self.printMessage("Epoch " + str(epoch) + " finished at ")
            
            cost = self.computeMultipleCost()
            print ("cost for this run was ", cost)
            self.runningCost.append(cost)
            
        
    def init (self):
        
        sampleX = self.X[0:64,:,:,:]
        wscale=1e-2
        self.params = self.init_params(self.conv_spec_list, self.dense_channels)
    
        self.params = self.init_dense_from_sample(self.params, self.conv_stack_forward, sampleX, self.conv_spec_list, self.dense_channels, wscale )
        self.optimizer = ao.AdamOptimizer(self.lr, self.weight_decay)

    def run(self): 
        inTraining = True
        self.train_cnn(self.X, self.Y , self.params, self.batchSize, self.epochs, inTraining)
        
    def testTraining(self):
        
        inTraining = False
        self.dropOutByLayer = [False, False, False]
        self.train_cnn(self.X, self.Y , self.params, self.batchSize , 1 , inTraining)
        acc = self.accuracy(self.Y_batch_results)
        cost = self.computeMultipleCost()
       
        print ("Training Accuracy was " + str( int(acc)) + " with cost " + str (cost))

    def test (self,  Xtest, Ytest):
        
        self.X = Xtest
        self.Y = Ytest
        epochs = 1 
        inTraining = False
        self.dropOutByLayer = [False, False, False]
        self.augmentation = False
        self.train_cnn(self.X, self.Y , self.params, self.batchSize , epochs, inTraining)
        acc = self.accuracy(self.Y_batch_results)
        cost = self.computeMultipleCost()
       
        print ("Test Accuracy was " + str( int(acc)) + " with cost " + str (cost))

    def accuracy(self, scores):
      
        # get the indices of the maximum in each column
        labelIndices = np.argmax(self.Y, axis = 1)
        scoresIndices = np.argmax(scores, axis = 1)
        acc = np.mean(labelIndices == scoresIndices)
        return acc*100      

    def save_params(self):
    
        #ex1, ex2 = self.optimizer.getCheckSums()
        #mess = " check sums are " + str(ex1) + " and " + str(ex2)
        #print ( mess)
        self.save_checkpoint("model.pkl", self.params, self.optimizer)
        
    def save_params_by_name(self, filename):
        self.save_checkpoint(filename, self.params, self.optimizer)
    

    def load_params(self, filename="model.pkl"):
       
        params, optimizer = self.load_checkpoint(filename, ao.AdamOptimizer)
        self.optimizer = optimizer
        self.params = params
        #ex1, ex2 = self.optimizer.getCheckSums()
        #mess = " check sums are " + str(ex1) + " and " + str(ex2)
        #print (mess)

        
    def load_checkpoint(self, path, optimizer_class):

            with open(path, "rb") as f:
                checkpoint = pickle.load(f)

            saved_params = checkpoint["params"]
            params = saved_params

            # -------- RESTORE OPTIMIZER --------
            opt_state = checkpoint["optimizer"]

            optimizer = optimizer_class(
                lr=opt_state["lr"],
                beta1=opt_state["beta1"],
                beta2=opt_state["beta2"],
                eps=opt_state["eps"],
                weight_decay=opt_state["weight_decay"]
            )

            optimizer.t = opt_state["t"]
            optimizer.m = opt_state["m"]
            optimizer.v = opt_state["v"]

            extra = checkpoint.get("extra", None)
            #extra = checkpoint["extra"]
            #ex1 = extra["ex1"]
            #ex2 = extra["ex2"]

            print(f"✅ Checkpoint loaded from: {path}")

            return params, optimizer

    def save_checkpoint(self, path, params, optimizer,extra= None):
        """
        path: filename, e.g. 'checkpoint.pkl'
        params: dict of all model parameters
        optimizer: AdamOptimizer instance
        extra: optional dict (epoch, accuracy, etc)
        """
        checkpoint = {
            "params": params,

            "optimizer": {
                "lr": optimizer.lr,
                "beta1": optimizer.beta1,
                "beta2": optimizer.beta2,
                "eps": optimizer.eps,
                "weight_decay": optimizer.weight_decay,
                "t": optimizer.t,
                "m": optimizer.m,
                "v": optimizer.v
            },
            
            "extra": extra

        }

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"✅ Checkpoint saved to: {path}")

    def random_horizontal_flip(self,images, p=0.5):
            """
            images: (N, C, H, W)
            """
            N = images.shape[0]
            flipped = images.copy()

            for i in range(N):
                if np.random.rand() < p:
                    flipped[i] = flipped[i, :, :, ::-1]

            return flipped

    def random_crop(self, images, crop_size=32, padding=4):
        """
        images: (N, C, H, W)
        """
        N, C, H, W = images.shape
        padded = np.pad(
            images,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )

        cropped = np.zeros((N, C, crop_size, crop_size), dtype=images.dtype)

        for i in range(N):
            y = np.random.randint(0, H + 2 * padding - crop_size + 1)
            x = np.random.randint(0, W + 2 * padding - crop_size + 1)
            cropped[i] = padded[i, :, y:y+crop_size, x:x+crop_size]

        return cropped
    
    def cutout_batch(self, x, size=8, p=0.5):
        """
        Apply Cutout to a batch of images.

        Inputs:
        - x: input batch of shape (N, C, H, W)
        - size: cutout square size
        - p: probability of applying cutout per image

        Returns:
        - x_out: augmented batch
        """
        N, C, H, W = x.shape
        x_out = x.copy()

        for i in range(N):
            if np.random.rand() > p:
                continue

            y = np.random.randint(0, H)
            x0 = np.random.randint(0, W)

            y1 = np.clip(y - size // 2, 0, H)
            y2 = np.clip(y + size // 2, 0, H)
            x1 = np.clip(x0 - size // 2, 0, W)
            x2 = np.clip(x0 + size // 2, 0, W)

            x_out[i, :, y1:y2, x1:x2] = 0.0

        return x_out
    
    def augment_cifar10(self, images):
        """
        images: (N, C, 32, 32)
        """
        images = self.random_crop(images, crop_size=32, padding=4)
        images = self.random_horizontal_flip(images, p=0.5)
       
        return images

    def getCosineLr(self, current_step):
        """
        Calculates learning rate at a given step using cosine decay.
        """
        total_steps = self.totalEpochs
        lr_max = self.maxLr
        lr_min = self.minLr
        # Ensure current_step does not exceed total_steps
        current_step = min(current_step, total_steps)
        
        # Cosine decay formula
        cosine_decay = 0.5 * (1 + mth.cos(mth.pi * current_step / total_steps))
        # Scale and shift to [lr_min, lr_max]
        return lr_min + (lr_max - lr_min) * cosine_decay
    
        # Example: Get LR for step 50 of 100
        # current_lr = get_cosine_lr(50, 100, lr_max=0.1, lr_min=0.001)
        # print(f"Learning Rate at step 50: {current_lr}")    
        
    def gapForward(self, x):
        """
        Global Average Pooling forward pass.

        Inputs:
        - x: Input data of shape (N, C, H, W)

        Returns a tuple of:
        - out: Output data of shape (N, C)
        - cache: Cached values for backward pass
        """
        N, C, H, W = x.shape

        # Mean over spatial dimensions
        out = x.mean(axis=(2, 3))

        #cache = (x.shape,)
        #return out, cache
        return out

    
    def gapBackward(self, dout, cache):
        """
        Global Average Pooling backward pass.

        Inputs:
        - dout: Upstream gradients of shape (N, C)
        - cache: Cached values from forward pass

        Returns:
        - dx: Gradient with respect to input x, shape (N, C, H, W)
        """
       
        (N, C, H, W,) = cache
        
        dx = dout[:, :, None, None] / (H * W)
        dx = np.broadcast_to(dx, (N, C, H, W))

        return dx
    
    def printMessage(self, mess):
        time =   datetime.now().time()    
        current = time.replace(microsecond=0)
        print (mess ,current)
       
