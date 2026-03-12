import numpy as np

class ActivationFunctions:
    def __init__(self, name = "sigmoid"):
        self.name = name
    
    @property
    def get_info_actifation_pairing(self):
        return {
            'unit_step': {
                1: self.unit_step,
                0: False,
                -1: self.unit_step_deriv,
            },
            
            'sign_step': {
                1: self.sign,
                0: False,
                -1: self.sign_deriv,
            },
            
            'piece_wise_linear': {
                1: self.piece_wise_linear,
                0: True,
                -1: self.piece_wise_linear_deriv,
            },
            
            'sigmoid': {
                1: self.sigmoid,
                0: True,
                -1: self.sigmoid_deriv,
            },
            
            'hyperbolic_tanh': {
                1: self.hyperbolic_tanh,
                0: True,
                -1: self.hyperbolic_tanh_deriv,
            },
            
            'relu': {
                1: self.relu,
                0: True,
                -1: self.relu_deriv,
            },
            
            'softplus': {
                1: self.softplus,
                0: True,
                -1: self.softplus_deriv,
            },
            
            'leaky_relu': {
                1: self.leaky_relu,
                0: True,
                -1: self.leaky_relu_deriv,
            },
        }
    
    def unit_step(self, z):
        return np.where(z < 0, 0, np.where(z > 0, 1, 0.5))
    
    # doesn't exist in dl
    def unit_step_deriv(self, z):
        pass
    
    def sign(self, z):
        return np.where(z < 0, -1, np.where(z > 0, 1, 0))
    
    # doesn't exist in dl
    def sign_deriv(self, z):
        pass
    
    def piece_wise_linear(self, z):
        return np.where(
            z >= 0.5,
            1,
            np.where(
                z <= -0.5,
                0,
                z + 0.5
            )
        )
    
    def piece_wise_linear_deriv(self, z):
        return np.where(
            (z > -0.5) & (z < 0.5),
            1,
            0
        )

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_deriv(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def hyperbolic_tanh(self, z):
        return np.tanh(z)
    
    def hyperbolic_tanh_deriv(self, z):
        t = np.tanh(z)
        return 1 - t**2
    
    def relu(self, z):
        return (z > 0).astype(float)
    
    def relu_deriv(self, z):
        return np.where(z > 0, 1, 0)
    def leaky_relu(self, z):
        return np.where(z > 0, z, 0.01 * z)

    def leaky_relu_deriv(self, z):
        return np.where(z > 0, 1, 0.01)

    def softplus(self, z):
        return np.log1p(np.exp(z))
    
    def softplus_deriv(self, z):
        return 1 / (1 + np.exp(-z))
    