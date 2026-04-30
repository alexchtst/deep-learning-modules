import numpy as np
from modules.base_model import DeepLearningBaseModel
from modules.nonlinear_function import PairingFunction

class NeuralNetworkTrainable(DeepLearningBaseModel):
    def __init__(
        self,
        input_size=8,
        output_size=8,
        activation_func="sigmoid",
        weights=None,
        bias=None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        
        super().__init__(name="neural", activation_function_key=activation_func)
        
        if weights is not None:
            self.W = weights
        else:
            self.__weights__ = np.random.randn(self.input_size, self.output_size) * 0.1
        
        if bias is not None:
            self.b = bias
        else:
            self.__bias__ = np.random.randn(1, self.__output_size__) * 0.1
        
        self.pair_function = PairingFunction(activation_func)
        self.__cache__ = {
            'x': None,
            'a': None,
            'z': None,
            'w': self.W,
            'b': self.b,
            'dl': 0
        }
        
    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        a = self.pair_function.forward(z)

        self.__cache__['x'] = x
        self.__cache__['z'] = z
        self.__cache__['a'] = a

        return a
    
    def backward(self, prev_delta):

        x = self.__cache__['x']
        z = self.__cache__['z']
        W = self.W

        da_dz = self.pair_function.derivative(z)
        delta = prev_delta * da_dz
        dl_dw = (x.T @ delta) / x.shape[0]
        dl_db = np.sum(delta, axis=0, keepdims=True) / x.shape[0]
        delta_prev = delta @ W.T

        return dl_dw, dl_db, delta_prev
    
    def update_step(self, dl_dw, dl_db, lr):

        self.W -= lr * dl_dw
        self.b -= lr * dl_db

        self.__cache__['w'] = self.W
        self.__cache__['b'] = self.b