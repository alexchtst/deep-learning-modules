class ConvTranspose2DTrainable:
    def __init__(self):
        pass
    
    @property
    def get_name(self):
        pass
    
    @property
    def get_cache(self):
        pass
    
    @property
    def get_weights(self):
        pass
    
    @property
    def get_bias(self):
        pass
    
    @property
    def get_params(self):
        pass
    
    def forward(self, x):
        pass
    
    def backward(self, prev_delta):
        pass
    
    def update_step(self, dl_dw, dl_db, lr):
        pass
    