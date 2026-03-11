import numpy as np
from  modules.v1.activation_function import ActivationFunctions

class NeuralNetworkTrainable:
    
    def __init__(
        self, 
        input_size=8,
        output_size=8,
        activation_func="sigmoid",
        weights=None,
        bias=None,
    ):
        
        self.__mod_name__ = "neural"

        self.activation_name = activation_func

        act = ActivationFunctions(activation_func)
        self.activation_pair = act.get_info_actifation_pairing[activation_func]

        self.activation = self.activation_pair[1]
        self.activation_deriv = self.activation_pair[-1]

        # params initialization
        if weights is not None:
            self.__weights__ = weights
            self.__input_size__ = weights.shape[0]
            self.__output_size__ = weights.shape[1]
        else:
            self.__input_size__ = input_size
            self.__output_size__ = output_size
            self.__weights__ = np.random.randn(self.__input_size__, self.__output_size__) * 0.1

        if bias is not None:
            self.__bias__ = bias
        else:
            self.__bias__ = np.random.randn(1, self.__output_size__) * 0.1

        self.__cache__ = {
            'x': None,
            'a': None,
            'z': None,
            'w': self.__weights__,
            'b': self.__bias__,
            'dl': 0
        }
        
    @property
    def get_name(self):
        return self.__mod_name__
    
    @property
    def get_cache(self):
        return self.__cache__
    
    @property
    def get_weights(self):
        return self.__cache__['w']
    
    @property
    def get_bias(self):
        return self.__cache__['b']
    
    @property
    def get_params(self):
        return self.__cache__['w'], self.__cache__['b']
    
    def forward(self, x):

        z = np.dot(x, self.__weights__) + self.__bias__
        a = self.activation(z)

        self.__cache__['x'] = x
        self.__cache__['z'] = z
        self.__cache__['a'] = a

        return a
    
    def backward(self, prev_delta):

        x = self.__cache__['x']
        z = self.__cache__['z']
        W = self.__weights__

        da_dz = self.activation_deriv(z)

        # delta layer ini
        delta = prev_delta * da_dz

        # gradient parameter
        dl_dw = x.T @ delta
        # dl_dw = (x.T @ delta) / x.shape[0]
        dl_db = np.sum(delta, axis=0, keepdims=True)
        # dl_db = np.sum(delta, axis=0, keepdims=True) / x.shape[0]

        # delta untuk layer sebelumnya
        delta_prev = delta @ W.T

        return dl_dw, dl_db, delta_prev

    def update_step(self, dl_dw, dl_db, lr):

        self.__weights__ -= lr * dl_dw
        self.__bias__ -= lr * dl_db

        self.__cache__['w'] = self.__weights__
        self.__cache__['b'] = self.__bias__