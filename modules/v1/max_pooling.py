import numpy as np

class MaxPool2DTrainable:
    def __init__(
        self,
        kernel_size=2,
    ):
        self.__mod_name__ = "maxpool"
        self.k_size = kernel_size
        self.strd = kernel_size

        self.__cache__ = {
            'x': None,
            'mask': None,
        }

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
        in_channel, H, W = x.shape
        k = self.k_size

        H_out = H // k
        W_out = W // k

        out = np.zeros((in_channel, H_out, W_out))

        mask = np.zeros_like(x, dtype=bool)

        for c in range(in_channel):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.strd
                    h_end   = h_start + k
                    w_start = j * self.strd
                    w_end   = w_start + k

                    patch = x[c, h_start:h_end, w_start:w_end]

                    max_val = np.max(patch)
                    out[c, i, j] = max_val

                    local_mask = (patch == max_val)
                    local_mask[np.unravel_index(np.argmax(local_mask), local_mask.shape)] = True
                    local_mask_unique = np.zeros_like(local_mask)
                    local_mask_unique[np.unravel_index(np.argmax(patch), patch.shape)] = True

                    mask[c, h_start:h_end, w_start:w_end] = local_mask_unique

        self.__cache__['x'] = x
        self.__cache__['mask'] = mask

        return out

    def backward(self, prev_delta):
        x = self.__cache__['x']
        mask = self.__cache__['mask']

        in_channel, H, W = x.shape
        k = self.k_size

        H_out = H // k
        W_out = W // k

        delta_prev = np.zeros_like(x)

        for c in range(in_channel):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.strd
                    h_end   = h_start + k
                    w_start = j * self.strd
                    w_end   = w_start + k

                    delta_prev[c, h_start:h_end, w_start:w_end] = (
                        mask[c, h_start:h_end, w_start:w_end] * prev_delta[c, i, j]
                    )

        return None, None, delta_prev

    def update_step(self, dl_dw, dl_db, lr):
        pass
