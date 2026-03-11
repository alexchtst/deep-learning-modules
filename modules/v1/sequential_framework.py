import numpy as np

class SeqFramework:
    def __init__(self):
        
        """ __models_conf__ is a list that contain of
        {
            name: "conv2d" | "neural" | "pooling" | "conv2dtranspose",
            params: {
                "weights": [.....],
                "bias": [.....],
            },
            index: 0,
            hasforward: True | False
        }
        """
        self.__models_conf__ = []
        self.__models__ = []
        
        self.__dl_dw__ = []
        self.__dl_db__ = []
        self.__loss_hist__ = []
    
    @property
    def loss_hist(self):
        return self.__loss_hist__
    
    @property
    def get_models(self):
        return self.__models__
    
    @property
    def get_models_config(self):
        return self.__models_conf__
    
    def __create_template_cofig__(self, name, w, b, idx, hasforward = True):
        return {
            "name": name,
            "params": {
                "weights": w,
                "bias": b,
            },
            "index": idx,
            "hasforward": hasforward
        }
        
    def add(self, model, hasforward=True):
        self.__models__.append(model)
        last_idx = len(self.__models__)
        conf = self.__create_template_cofig__(model.get_name, model.get_weights, model.get_bias, last_idx, hasforward)
        self.__models_conf__.append(conf)
        
    def run_forward(self, x):
        out = x
        for model in self.__models__:
            out = model.forward(out)
        return out
    
    def calculate_delta(self, y_pred, y_true):
        delta = y_pred - y_true
        return delta
    
    def run_backprop_and_update(self, y_pred, y_true, lr=0.01):

        delta = self.calculate_delta(y_pred, y_true)

        for model in reversed(self.__models__):

            dl_dw, dl_db, delta = model.backward(delta)

            model.update_step(dl_dw, dl_db, lr)
    
    def train(self, x, Y, epochs=10, lr=0.01, batch_size=32):
        rng = np.random.default_rng()
        num_samples = x.shape[0] // batch_size
        
        for epoch in range(epochs):

            idx = rng.choice(x.shape[0], size=num_samples, replace=True)
            x_input = x[idx]
            y_true = Y[idx]

            y_pred = self.run_forward(x_input)
            loss = np.mean((y_pred - y_true) ** 2) / 2
            print(f"epoch {epoch} loss:", loss)

            self.__loss_hist__.append(loss)
            self.run_backprop_and_update(y_pred, y_true, lr)
            
    def predict(self, x):
        return self.run_forward(x)