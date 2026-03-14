class FlattenMatrix:
    def __init__(self):
        self.__mod_name__ = "flatten"
        self.__cache__ = {'x_shape': None}

    @property
    def get_name(self):
        return self.__mod_name__

    @property
    def get_cache(self):
        return self.__cache__

    @property
    def get_weights(self):
        return None

    @property
    def get_bias(self):
        return None

    @property
    def get_params(self):
        return None, None

    def forward(self, x):
        self.__cache__['x_shape'] = x.shape
        return x.flatten().reshape(1, -1)

    def backward(self, prev_delta):
        x_shape = self.__cache__['x_shape']
        delta_prev = prev_delta.reshape(x_shape)
        return None, None, delta_prev

    def update_step(self, dl_dw, dl_db, lr):
        pass