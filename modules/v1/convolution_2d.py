from  modules.v1.activation_function import ActivationFunctions
import numpy as np

class Conv2DTrainable:
    def __init__(
        self, 
        in_channel=3,
        out_channel=3,
        activation_func="sigmoid",
        kernel_size=3,
    ):
        self.__mod_name__ = "conv2d"
        
        self.out_channel = out_channel
        self.in_channel = in_channel
        
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
        self.__batch_cache__ = []
        
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
    
    def get_index_map(self, x):
        c_in, h_in, w_in = x.shape
        num_f_k, c_k, h_k, w_k = self.__weights__.shape
        
        if h_in < h_k or w_in < w_k:
            raise ValueError(
                f"Kernel size ({h_k},{w_k}) lebih besar dari input ({h_in},{w_in})"
            )
        
        max_h_index = h_in - h_k + 1
        max_w_index = w_in - w_k + 1
        
        return max_h_index, max_w_index

    def conv_2d(self, x):
        max_h_index, max_w_index = self.get_index_map(x)

        out = np.zeros((self.out_channel, max_h_index, max_w_index))

        for f in range(self.out_channel):
            kernel = self.__weights__[f]

            for row in range(max_h_index):
                for col in range(max_w_index):

                    patch = x[:, row:row+self.k_size, col:col+self.k_size]

                    out[f, row, col] = np.sum(
                        patch * kernel
                    ) + self.__bias__[f]

        return out
    
    def __single_forward(self, x):
        z = self.conv_2d(x)
        a = self.activation(z)
        
        
        self.__cache__['z'] = z
        self.__cache__['a'] = a
        
        return a
    
    def forward(self, x):
        B, C, H, W = x.shape

        outputs = []
        self.__batch_cache__ = []

        for i in range(B):
            single_x = x[i]

            self.__cache__['x'] = single_x

            a = self.__single_forward(single_x)

            self.__batch_cache__.append({
                'x': single_x,
                'z': self.__cache__['z'],
                'a': self.__cache__['a']
            })

            outputs.append(a)

        return np.stack(outputs, axis=0)
    
    def backward(self, prev_delta):
       
        B = prev_delta.shape[0]

        total_dl_dw = np.zeros_like(self.__weights__)
        total_dl_db = np.zeros_like(self.__bias__)

        delta_prev_batch = []

        for i in range(B):

            cache = self.__batch_cache__[i]

            self.__cache__['x'] = cache['x']
            self.__cache__['z'] = cache['z']

            dl_dw, dl_db, delta_prev = self.__single_backward(
                prev_delta[i]
            )

            total_dl_dw += dl_dw
            total_dl_db += dl_db
            delta_prev_batch.append(delta_prev)

        total_dl_dw /= B
        total_dl_db /= B

        self.__cache__['dl'] = (
            total_dl_dw,
            total_dl_db
        )

        return total_dl_dw, total_dl_db, np.stack(delta_prev_batch, axis=0)
    
    def __single_backward(self, prev_delta):

        x = self.__cache__['x']
        z = self.__cache__['z']
        W = self.__weights__

        C_out, C_in, K, _ = W.shape
        C_in, H_in, W_in = x.shape

        H_out = H_in - K + 1
        W_out = W_in - K + 1

        da_dz = self.activation_deriv(z)

        # delta layer ini
        delta = prev_delta * da_dz

        # gradient weight
        dl_dw = np.zeros_like(W)

        for f in range(C_out):
            for c in range(C_in):
                for i in range(K):
                    for j in range(K):

                        patch = x[c, i:i+H_out, j:j+W_out]
                        dl_dw[f, c, i, j] = np.sum(
                            patch * delta[f]
                        )

        # gradient bias
        dl_db = np.sum(delta, axis=(1,2))

        # delta untuk layer sebelumnya
        delta_prev = np.zeros_like(x)

        for f in range(C_out):
            for row in range(H_out):
                for col in range(W_out):

                    delta_prev[:, row:row+K, col:col+K] += (
                        W[f] * delta[f, row, col]
                    )

        return dl_dw, dl_db, delta_prev

    def update_step(self, dl_dw, dl_db, lr):
        self.__weights__ -= lr * dl_dw
        self.__bias__ -= lr * dl_db
        self.__cache__['w'] = self.__weights__
        self.__cache__['b'] = self.__bias__

class Conv2d:
    def __init__(self, kernel_weights, bias):
        self.kernel = kernel_weights
        self.bias = bias
        
    def conv_2d(self, x):
        pass
    
    def pass_forward(self, x):
        pass