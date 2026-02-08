
import numpy as np
import pickle 
from numpy import linalg as LA
"""
grads = {
        'W_conv': dW_conv,
        'b_conv': db_conv,
        'gamma': dgamma,
        'beta': dbeta,
        'W_fc': dW_fc,
        'b_fc': db_fc
    }
   # Initialize weights
params = {
    'W_conv': np.random.randn(4, 3, 3, 3) * 0.1,
    'b_conv': np.zeros(4),
    'gamma': np.ones(4),
    'beta': np.zeros(4),
    'W_fc': np.random.randn(4 * 4 * 4, num_classes) * 0.1,
    'b_fc': np.zeros(num_classes)
}
 
    
"""

class AdamOptimizer: 
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}
        
  
    def step(self, params, grads):
        self.t += 1

        for k in params.keys():

            if k not in grads:
                continue

            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])

            g = grads[k]

            # Adam moments
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            # AdamW weight decay (decoupled)
            if 'W' in k:
                params[k] *= (1 - self.lr * self.weight_decay)

            # Parameter update
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return params


   
    def save_state(self, filename="model_optimizer.npz"):
        flat = {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "t": self.t
        }

        # Save m and v buffers
        for k, v in self.m.items():
            flat[f"m_{k}"] = v
        for k, v in self.v.items():
            flat[f"v_{k}"] = v

        np.savez(filename, **flat)
        
        print(f"✅ Optimizer state saved to {filename}")
        

    def load_state(self, filename="model_optimizer.npz"):
        data = np.load(filename)

        # Restore scalars
        self.lr = float(data["lr"])
        self.beta1 = float(data["beta1"])
        self.beta2 = float(data["beta2"])
        self.eps = float(data["eps"])
        self.weight_decay = float(data["weight_decay"])
        self.t = int(data["t"])      


        # Restore m and v buffers
        for k in self.params.keys():
            mk = f"m_{k}"
            vk = f"v_{k}"
            if mk in data:
                self.m[k] = data[mk]
            if vk in data:
                self.v[k] = data[vk]
        

        print("✅ Optimizer state loaded successfully")
        


    def save_checkpoint(path, params, optimizer, extra=None):
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
            
    def getCheckSums(self):
        ex1 = self.lr + self.beta1 + self.beta2 + self.eps + self.weight_decay + self.t
        ex2 = 0.0
        for key, value in self.m.items():
            ex2 += np.sum(value)
        return ex1, ex2
    
        #   to take the mean gradient per layer I would add the norms for each layer 
        #   and average (divide by 3 in my case)

    def printFrobeniusNorm(self, grads, params):
         
        substrings = ["W_dens"]
        
        for k in params.keys():
            if any(sub in k for sub in substrings):
                if k in grads:
                    g = grads[k]
                    mess = " gradient for " +  str(k) + " is "
                    print (mess, np.linalg.norm(g)) 

