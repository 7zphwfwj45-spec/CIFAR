import sys
sys.path.append("/Volumes/MyDrive/sharedPythonFiles")
import numpy as np
import AdamOptimizer as ao
from datetime import datetime
import pickle
class Control:
    def __init__(self, batchSize, lr , epochs, dropOutRate):
        
        self.X = 0
        self.Y = 0
        self.batchSize = batchSize
        self.lr = lr
        self.epochs = epochs
        
        self.dropOut = False
        if dropOutRate > 0.0:
            self.dropOut = True
        self.dropOutRate = dropOutRate
        self.Y_batch_results = 0.0
        self.params = {}
        self.L2Lambda = 0.0
        self.weight_decay=1e-4
        self.loss = []
        print (" Created Convolution Class")
        
        self.conv_spec_list = [
            {'filter_size': 3, 'in_ch': 3, 'out_ch': 16, 'stride': 1, 'pad': 1, 'use_bn': True,
            'pool_param': {'pool_height': 2, 'pool_width': 2, 'stride': 2}},
            {'filter_size': 3, 'in_ch': 16, 'out_ch': 32, 'stride': 1, 'pad': 1, 'use_bn': True,
            'pool_param': {'pool_height': 2, 'pool_width': 2, 'stride': 2}},
            {'filter_size': 3, 'in_ch': 32, 'out_ch': 32, 'stride': 1, 'pad': 1, 'use_bn': True,
            'pool_param': {'pool_height': 2, 'pool_width': 2, 'stride': 2}}
        ]
        self.dense_channels = [128, 64, 10]  
        
    def setTrainingData(self, X,Y):
        self.X = X
        self.Y = Y
        self.Y_batch_results = np.zeros(Y.shape)
        

    def setDropout(self, dropOutRate):
        self.dropOut = False
        if dropOutRate > 0.0:
            self.dropOut = True
            
    def setLearningRate(self, lr):
        self.lr = lr
        
# delete a instantioted class instance.
    def __del__(self):
        print ("f{self.name} destroyed")     
    
    def conv_forward(self, x, w, b, stride=1, pad=1):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape

        H_out = 1 + (H + 2*pad - HH) // stride
        W_out = 1 + (W + 2*pad - WW) // stride

        x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
        out = np.zeros((N, F, H_out, W_out))

        for n in range(N):
            for f in range(F):
                for i in range(0, H_out):
                    for j in range(0, W_out):
                        h_start, w_start = i*stride, j*stride
                        x_slice = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                        out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]

        cache = (x, w, b, stride, pad, x_padded)
        return out, cache

 #      backward expectsx, x_hat, mu, var, gamma, beta, eps = cache
 
    def batchnorm_forward2(self, x, gamma, beta, running_mean, running_var,
                        momentum=0.9, eps=1e-5, training=True):
        """
        x : (N, C, H, W)
        gamma, beta : (C, 1)
        running_mean, running_var : (C, 1)
        """

        N, C, H, W = x.shape

        # ---- FLATTEN FOR CHANNEL-WISE BN ----
        x_flat = x.transpose(1, 0, 2, 3).reshape(C, -1)  # (C, N*H*W)

        # why do I care if I am in training ?
    
        mu  = x_flat.mean(axis=1, keepdims=True)      # (C,1)
        var = x_flat.var(axis=1, keepdims=True)       # (C,1)

            # Running updates
        running_mean[:] = momentum * running_mean + (1 - momentum) * mu
        running_var[:]  = momentum * running_var  + (1 - momentum) * var
        #else:
        #    mu = running_mean
        #    var = running_var

        std = np.sqrt(var + eps)                           # (C,1)
        #mean_w = np.mean(W)
        #std_w = np.std(W)
        
        #print(f"mean={mean_w:.4f}, std={std_w:.4f}")

        x_hat = (x_flat - mu) / std                        # (C,M)

        # ---- SCALE + SHIFT ----
        out_flat = gamma * x_hat + beta                    # (C,M)

        # ---- RESHAPE BACK ----
        out = out_flat.reshape(C, N, H, W).transpose(1, 0, 2, 3)

        cache = (x_hat, mu, std, gamma, x_flat)
        return out, cache

    def batchnorm_backward2(self, dout, cache):
        """
        dout : (N, C, H, W)
        """

        x_hat, mu, std, gamma, x_flat = cache
        N, C, H, W = dout.shape

        dout_flat = dout.transpose(1, 0, 2, 3).reshape(C, -1)  # (C,M)
        M = dout_flat.shape[1]

        # ---- PARAM GRADS ----
        dbeta = dout_flat.sum(axis=1, keepdims=True)                  # (C,1)
        dgamma = np.sum(dout_flat * x_hat, axis=1, keepdims=True)      # (C,1)

        dxhat = dout_flat * gamma                                       # (C,M)

        dvar = np.sum(dxhat * (x_flat - mu) * -0.5 * std**(-3), axis=1, keepdims=True)
        dmu  = np.sum(dxhat * -1 / std, axis=1, keepdims=True) + \
            dvar * np.mean(-2 * (x_flat - mu), axis=1, keepdims=True)

        dx = (dxhat / std) + (dvar * 2 * (x_flat - mu) / M) + (dmu / M)

        dx = dx.reshape(C, N, H, W).transpose(1, 0, 2, 3)

        return dx, dgamma, dbeta



    def batchnorm_forward(self, x, gamma, beta, runMean, runVar , inTraining) :
        N, C, H, W = x.shape
        x_flat = x.transpose(1,0,2,3).reshape(C, -1)
        eps=1e-5
        momentum = 0.9
        
        if inTraining == True:
            mu = np.mean(x_flat, axis=1,  keepdims=True)
            var = np.var(x_flat, axis=1, keepdims=True)
            
            runMean[:] = momentum * runMean + (1 - momentum)*mu
            runVar[:] = momentum * runVar + (1 - momentum)*var
        else:
            mu = runMean
            var = runVar

        std = np.sqrt(var + eps)
        x_hat = (x_flat - mu) /std
        #out_flat = gamma[:, None]*x_hat + beta[:, None] -> None must be removed - done 
        out_flat = gamma*x_hat + beta
        out = out_flat.reshape(C, N, H, W).transpose(1,0,2,3) 

        cache = (x, x_hat, mu, var, gamma, beta, eps)
        return out, cache, runMean,runVar

    def maxpool_forward_faster(x, pool_size=2, stride=2):
        N, C, H, W = x.shape
        H_out = (H - pool_size)//stride + 1
        W_out = (W - pool_size)//stride + 1
        out = np.zeros((N, C, H_out, W_out))
        mask = np.zeros_like(x, dtype=bool)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size

                window = x[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(window, axis=(2,3), keepdims=True)
                out[:, :, i, j] = max_vals.squeeze()
                mask[:, :, h_start:h_end, w_start:w_end] = (window == max_vals)
        cache = (x, pool_size, stride, mask)
        return out, cache

    def maxpool_forward(self, x, pool_size=2, stride=2):
            N, C, H, W = x.shape
            H_out = (H - pool_size)//stride + 1
            W_out = (W - pool_size)//stride + 1
            out = np.zeros((N, C, H_out, W_out))
            mask = np.zeros_like(x)

            for n in range(N):
                for c in range(C):
                    for i in range(H_out):
                        for j in range(W_out):
                            h_start, w_start = i*stride, j*stride
                            patch = x[n,c,h_start:h_start+pool_size,w_start:w_start+pool_size]
                            m = np.max(patch)
                            out[n,c,i,j] = m
                            mask[n,c,h_start:h_start+pool_size,w_start:w_start+pool_size] = (patch == m)
            cache = (x, mask, pool_size, stride)
            return out, cache
    

    def dense_batchnorm_forward(self, x, gamma, beta, eps=1e-5):
        # x: (N, H)
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        x_hat = (x - mu) / np.sqrt(var + eps)
        out = gamma * x_hat + beta

        cache = (x, x_hat, mu, var, gamma, beta, eps)
        return out, cache
 
    def dense_batchnorm_backward(self, dout, cache):
        x, x_hat, mu, var, gamma, beta, eps = cache
        N, D = x.shape

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * x_hat, axis=0)

        dxhat = dout * gamma
        dvar = np.sum(dxhat * (x - mu) * -0.5 * (var + eps)**(-1.5), axis=0)
        dmu = np.sum(dxhat * -1 / np.sqrt(var + eps), axis=0) \
            + dvar * np.mean(-2 * (x - mu), axis=0)

        dx = dxhat / np.sqrt(var + eps) \
            + dvar * 2 * (x - mu) / N \
            + dmu / N

        return dx, dgamma, dbeta


    def relu_forward(self, x):
        out = np.maximum(0, x)
        cache = x
        return out, cache

    def relu_backward(self, dout, cache):
        x = cache
        dx = dout * (x > 0)
        return dx

    
    def batchnorm_backward(self,dout, cache):
            x, x_hat, mu, var, gamma, beta, eps = cache
            N, C, H, W = x.shape
            x_flat = x.transpose(1,0,2,3).reshape(C, -1)
            dout_flat = dout.transpose(1,0,2,3).reshape(C, -1)

            m = dout_flat.shape[1]
            # dxhat = dout_flat * gamma[:, None]- remove None 
            dxhat = dout_flat * gamma
            dvar = np.sum(dxhat * (x_flat - mu) * -0.5 * (var + eps)**(-1.5), axis=1, keepdims=True)
            dmu = np.sum(dxhat * -1/np.sqrt(var + eps), axis=1, keepdims=True) + dvar * np.mean(-2*(x_flat - mu), axis=1, keepdims=True)
            dx_flat = dxhat / np.sqrt(var + eps) + dvar * 2*(x_flat - mu)/m + dmu/m

            dx = dx_flat.reshape(C,N,H,W).transpose(1,0,2,3)
            dgamma = np.sum(dout * x_hat.reshape(C,N,H,W).transpose(1,0,2,3), axis=(0,2,3))
            dbeta = np.sum(dout, axis=(0,2,3))
            return dx, dgamma, dbeta
    
    def maxpool_backward_faster(dout, cache):
        x, pool_size, stride, mask = cache
        N, C, H, W = x.shape
        H_out, W_out = dout.shape[2:]
        dx = np.zeros_like(x)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size

                dx[:, :, h_start:h_end, w_start:w_end] += (
                    mask[:, :, h_start:h_end, w_start:w_end] *
                    dout[:, :, i, j][:, :, None, None]
                )
        return dx

    def maxpool_backward(self, dout, cache):
        x, mask, pool_size, stride = cache
        N, C, H, W = x.shape
        H_out, W_out = dout.shape[2:]
        dx = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i*stride, j*stride
                        dx[n,c,h_start:h_start+pool_size,w_start:w_start+pool_size] += dout[n,c,i,j] * mask[n,c,h_start:h_start+pool_size,w_start:w_start+pool_size]
        return dx

    def conv_backward(self, dout, cache):
            x, w, b, stride, pad, x_padded = cache
            N, C, H, W = x.shape
            F, _, HH, WW = w.shape
            _, _, H_out, W_out = dout.shape

            dx_padded = np.zeros_like(x_padded)
            dw = np.zeros_like(w)
            db = np.sum(dout, axis=(0, 2, 3))

            for n in range(N):
                for f in range(F):
                    for i in range(H_out):
                        for j in range(W_out):
                            h_start, w_start = i*stride, j*stride
                            dx_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] += w[f] * dout[n, f, i, j]
                            dw[f] += x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] * dout[n, f, i, j]

            dx = dx_padded[:, :, pad:-pad, pad:-pad]
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

    
    def dropout_forward(self, x, p=0.5, training=True):
        """
        p: dropout probability (fraction to drop)
        """
        if not training or p == 0:
            return x, None
        
        mask = (np.random.rand(*x.shape) > p).astype(x.dtype)
        out = x * mask / (1 - p)
        cache = (mask, p, training)
        return out, cache


    def dropout_backward(self, dout, cache):
        mask, p, training = cache
        if not training or p == 0:
            return dout
        else:
            return dout * mask
        
    # x is (4,128) W is (128,10) b is (10,)
    def dense_forward(self,x, W, b):
        #N, C, H, W_ = x.shape
        #x_perm = x.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        #out = x_perm @ W + b                             # (N*H*W, C_out)
        #out = out.reshape(N, H, W_, -1).transpose(0, 3, 1, 2)
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
        for idx in range(len(dense_caches)-1, -1, -1):
            cache_affine, bn_cache, cache_relu, dropout_cache = dense_caches[idx]
        
            # dropout precceds relu
            if dropout_cache is not None:
                cur = self.dropout_backward(cur, dropout_cache)
                
            if cache_relu is not None:
                cur = self.relu_backward(cur, cache_relu)
                
            if bn_cache is not None:
                dz, dgamma, dbeta = self.dense_batchnorm_backward(cur, bn_cache)
                grads[f"gamma_dense_{idx+1}"] = dgamma
                grads[f"beta_dense_{idx+1}"] = dbeta
            else:
                dz = cur
                
            dx, dW, db = self.dense_backward(dz, cache_affine)
            grads[f"W_dense_{idx+1}"] = dW
            grads[f"b_dense_{idx+1}"] = db
            cur = dx

        # cur now has shape (N, C*H*W); reshape
        cur = cur.reshape(N, C, H, W)

        # Conv backward
        dx_conv_input, conv_grads = self.conv_stack_backward(cur, conv_caches, conv_spec_list)
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
            params[f"b_conv_{i}"] = np.zeros(C_out)
            if spec.get('use_bn', False):
                params[f"gamma_conv_{i}"] = np.ones((C_out,1))
                params[f"beta_conv_{i}"] = np.zeros((C_out,1))
                params[f"running_mean_{i}"] =  np.zeros((C_out,1))
                params[f"running_var_{i}"] =  np.ones((C_out,1))
                

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
        # Use the conv_forward helper defined below (conv_stack_forward) to get shape
       
        inTraining = False
        conv_out, _ = conv_stack_forward_fn(x_sample, params, conv_spec_list, inTraining)
        N, C_last, H_last, W_last = conv_out.shape
        flattened = C_last * H_last * W_last

        if "W_dense_1" not in params:
            params["W_dense_1"] = np.random.randn(flattened, dense_channels[0]) * wscale
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
            
            if spec.get('use_bn', False):
                
                out_bn, cache_bn= self.batchnorm_forward2(out_conv, 
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
                stride  = pool_param.get('stride')
                out_pool, cache_pool = self.maxpool_forward(out_relu, pool_size, stride)
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

        # flatten
        N, C, H, W = conv_out.shape
        flat = conv_out.reshape(N, C * H * W)
        caches['flatten_shape'] = (N, C, H, W)
        # dense forward
        dense_caches = []
        prev = flat
        num_dense = len(dense_channels)
        
        for i, dch in enumerate(dense_channels, start=1):
            
            Wd = params[f"W_dense_{i}"]
            bd = params[f"b_dense_{i}"]
            gamma = params[f"gamma_dense_{i}"]
            beta = params[f"beta_dense_{i}"]

            out_affine, cache_affine = self.dense_forward(prev, Wd, bd)
            if i < num_dense:
                out_bn, bn_cache = self.dense_batchnorm_forward(out_affine, gamma, beta)
                out_relu, cache_relu = self.relu_forward(out_bn)
                # this is broken needs much work
                if self.dropOut == True: 
                    # my cache is cache = (mask, p, training)
                    dropOut, dropout_cache = self.dropout_forward(out_relu, self.dropOutRate, inTraining)
                    prev = dropOut
                    dense_caches.append((cache_affine, bn_cache, cache_relu, dropout_cache))
                else:
                    prev = out_relu
                    dense_caches.append((cache_affine, bn_cache, cache_relu, None))
            else:
                dense_caches.append((cache_affine, None, None, None))
                prev = out_affine
        # 
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
                cur_grad, dgamma, dbeta = self.batchnorm_backward2(cur_grad, cache_bn)
                grads[f"gamma_conv_{i+1}"] = dgamma
                grads[f"beta_conv_{i+1}"] = dbeta

            # conv backward
            dx, dW, db = self.conv_backward(cur_grad, cache_conv)
            grads[f"W_conv_{i+1}"] = dW
            grads[f"b_conv_{i+1}"] = db

            cur_grad = dx

        return cur_grad, grads
    
    def softmax_backward_onehot (self, probs, Y):
        """
        Inputs:
        - probs: softmax output, shape (N, C)
        - Y: one-hot encoded true labels, shape (N, C)

        Returns:
        - dZ: gradient of loss wrt logits, shape (N, C)
        """

        N = probs.shape[0]
        return (probs - Y) / N

    def train_cnn(self, X, y, params, batch_size, epochs, inTraining):
        
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        for epoch in range(epochs):
           
            for b in range(num_batches):
                start = b * batch_size
                end = start + batch_size
                X_batch, y_batch = X[start:end], y[start:end]

                # scores is the ouput from the final layer.
                scores, cache = self.cnn_forward(X_batch, params, self.conv_spec_list, self.dense_channels, inTraining)  
                # dscores is the gradien wrt to the probabilities???
                loss, dscores, probs =  self.softmax_loss(scores, y_batch)
                
                if b%50 == 0:
                    loss = loss + self.l2_loss(params, self.weight_decay)
                    self.loss.append(loss)

        
                # store the results of this batch
                self.Y_batch_results[start:end, :] = probs
                
                if inTraining == True:
                    grads = self.cnn_backward(dscores, cache)
                    self.l2_grad(grads, params, self.weight_decay)
                     # Update
                    params = self.optimizer.step(grads)
                    
            acc = self.accuracy(self.Y_batch_results)
            cost = self.computeMultipleCost()
            if inTraining == True:
                print ("Accuracy was " + str( acc) + " with cost " + str (cost))
                #print (" At time ", datetime.now().time())
            else:
                print ("Test Accuracy was " + str( acc) + " with cost " + str (cost))
            
    def test_cnn(self, X, y):
         # scores is the ouput from the final layer.
        inTraining = False
        #scores, cache = self.cnn_forward(X, self.params, self.conv_spec_list, self.dense_channels, inTraining)  
        self.train_cnn(X, y, self.params, self.batchSize, 1 , inTraining)
        
    def init (self):
        
        sampleX = self.X[0:10,:,:,:]
        wscale=1e-2
        self.params = self.init_params(self.conv_spec_list, self.dense_channels)
    
        self.params = self.init_dense_from_sample(self.params, self.conv_stack_forward, sampleX, self.conv_spec_list, self.dense_channels, wscale )
        self.optimizer = ao.AdamOptimizer(self.params, self.lr, self.weight_decay)

    def run(self):
        
        inTraining = True
        self.train_cnn(self.X, self.Y , self.params, self.batchSize, self.epochs, inTraining)
        
    def testTraining (self):
       
        inTraining = False
        epochs = 1 
        # Critical: do not change the batch size.
        self.train_cnn(self.X, self.Y , self.params, self.batchSize , epochs,inTraining)

    def test (self,  Xtest, Ytest):
        
        self.X = Xtest
        self.Y = Ytest
        epochs = 1 
        inTraining = False
        self.dropOut = False
        self.Y_batch_results = np.zeros(self.Y.shape)
        self.train_cnn(self.X, self.Y , self.params, self.batchSize , epochs, inTraining)

    def accuracy(self, scores):
      
        # get the indices of the maximum in each column
        labelIndices = np.argmax(self.Y, axis = 1)
        scoresIndices = np.argmax(scores, axis = 1)
        
        # calculate the average where each element indicates 
        # whether the corresponding elements in labelIndices and myResultIndices are equal
        count = 0
        for ii in range(scores.shape[0]):
            if labelIndices[ii] == scoresIndices[ii]:
                count+= 1
        #print (" count was ", count)
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

            """
            # -------- SHAPE VALIDATION --------
            for k in params_template:
                if k not in saved_params:
                    raise ValueError(f"Missing parameter in checkpoint: {k}")

                if params_template[k].shape != saved_params[k].shape:
                    raise ValueError(
                        f"Shape mismatch for {k}: "
                        f"expected {params_template[k].shape}, "
                        f"got {saved_params[k].shape}"
                    )
            """
            # -------- RESTORE PARAMS --------
            params = saved_params

            # -------- RESTORE OPTIMIZER --------
            opt_state = checkpoint["optimizer"]

            optimizer = optimizer_class(
                params,
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

    def l2_loss(self,params, weight_decay):
        """
        Returns L2 penalty term:
            0.5 * λ * Σ ||W||^2
        """
        l2 = 0.0
        for name, W in params.items():
            if name.startswith("W"):          # No L2 on bias terms
                l2 += np.sum(W * W)
        return 0.5 * weight_decay * l2


    def l2_grad(self, grads, params, weight_decay):
        """
        Adds λW to gradients of weights.
        Mutates grads in-place.
        """
        for name in grads:
            if name.startswith("W"):
                grads[name] += weight_decay * params[name]
        

    """"
    THis is done for each mini batch - that is for each forward pass
    The matyrix is stored in the cache so it available for the backward pass
    During the test run drop[out is not in effect.
    For dense networks use 0.2 to 0.5 for the drop out rate

    	1.	Start modestly:
         Begin with 0.2 after activation layers. Thus the keep proability is 0.8
         if r is the dropput rate then p = 1 - r 
	    2.	Watch validation loss:
	    •	If validation loss > training loss (overfitting) → increase dropout.
	    •	If both training and validation loss are high (underfitting) → decrease dropout.
	    3.	Use different rates for different layers:
	    •	Early layers (close to input): 0.1–0.2
	    •	Deeper / dense layers: 0.3–0.5 

    

    # Forward
    Z = W @ X + b
    Z_norm, cache_bn = batchnorm_forward(Z, gamma, beta)
    A = relu(Z_norm)
    D = (np.random.rand(*A.shape) < p).astype(float)
    A_drop = (A * D) / p

    # Backward
    dA = (dA_drop * D) / p
    dZ_norm = dA * (Z_norm > 0)                 # ReLU backward
    dZ, dgamma, dbeta = batchnorm_backward(dZ_norm, cache_bn)
    dW = (dZ @ X.T) / m
    db = np.mean(dZ, axis=1, keepdims=True)
    dX = W.T @ dZ
    
    """