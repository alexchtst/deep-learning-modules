from  modules.v1.activation_function import ActivationFunctions
import numpy as np

class Conv2DTrainable:
    def __init__(
        self, 
        in_channel=3,
        out_channel=3,
        activation_func="sigmoid",
        kernel_size=3,
        strd=1,
        padding=0,
    ):
        self.__mod_name__ = "conv2d"
        self.strd = strd
        self.padding = padding
        
        self.activation_name = activation_func

        act = ActivationFunctions(activation_func)
        self.activation_pair = act.get_info_actifation_pairing[activation_func]
        self.activation = self.activation_pair[1]
        self.activation_deriv = self.activation_pair[-1]
        
        self.k_size = kernel_size
        fan_in = in_channel * self.k_size * self.k_size
        fan_out = out_channel * self.k_size * self.k_size

        limit = np.sqrt(6 / (fan_in + fan_out))
        self.__weights__ = np.random.uniform(
            -limit,
            limit,
            (out_channel, in_channel, self.k_size, self.k_size)
        )
        
        self.__bias__ = np.random.uniform(
            -limit,
            limit,
            (out_channel,)
        )
        
        self.__cache__ = {
            'x': None,
            'a': None,
            'z': None,
            'w': self.__weights__,
            'b': self.__bias__,
            'dl': 0,
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

    def conv_2d(self, x):
        in_channel, H, W = x.shape
        out_channel, _, k, _ = self.__weights__.shape
        
        # apply spatial padding
        if self.padding > 0:
            x = np.pad(
                x,
                pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
                constant_values=0
            )
            _, H, W = x.shape
        
        # output spatial
        H_out = (H - k) // self.strd + 1
        W_out = (W - k) // self.strd + 1
        
        z = np.zeros((out_channel, H_out, W_out))
        
        for f in range(out_channel):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.strd
                    h_end   = h_start + k
                    w_start = j * self.strd
                    w_end   = w_start + k
                    
                    patch = x[:, h_start:h_end, w_start:w_end]
                    z[f, i, j] = np.sum(patch * self.__weights__[f]) + self.__bias__[f]
        
        return z
    
    def forward(self, x):
        z = self.conv_2d(x)
        a = self.activation(z)
        
        self.__cache__['x'] = x
        self.__cache__['z'] = z
        self.__cache__['a'] = a
        
        return a
    
    def backward(self, prev_delta):
        x = self.__cache__['x_padded']
        z = self.__cache__['z']
        W = self.__weights__

        out_channel, in_channel, k, _ = W.shape
        _, H_out, W_out = z.shape

        da_dz = self.activation_deriv(z)

        delta = prev_delta * da_dz

        dl_dw = np.zeros_like(W)
        dl_db = np.zeros(out_channel)

        delta_prev_padded = np.zeros_like(x)

        for f in range(out_channel):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.strd
                    h_end   = h_start + k
                    w_start = j * self.strd
                    w_end   = w_start + k

                    patch = x[:, h_start:h_end, w_start:w_end]

                    dl_dw[f] += delta[f, i, j] * patch

                    delta_prev_padded[:, h_start:h_end, w_start:w_end] += delta[f, i, j] * W[f]

            dl_db[f] = np.sum(delta[f])

        if self.padding > 0:
            delta_prev = delta_prev_padded[
                :,
                self.padding:-self.padding,
                self.padding:-self.padding
            ]
        else:
            delta_prev = delta_prev_padded

        return dl_dw, dl_db, delta_prev

    def update_step(self, dl_dw, dl_db, lr):
        self.__weights__ -= lr * dl_dw
        self.__bias__ -= lr * dl_db
        self.__cache__['w'] = self.__weights__
        self.__cache__['b'] = self.__bias__
    