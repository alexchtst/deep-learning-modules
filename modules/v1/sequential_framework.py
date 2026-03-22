import numpy as np

class SeqFrameworkTrainable:
    def __init__(self):
        
        """ __models_conf is a list that contain of
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
        self.__models_conf = []
        self.__models = []
        
        self.__dl_dw = []
        self.__dl_db = []
        self.__loss_hist = []
    
    @property
    def loss_hist(self):
        return self.__loss_hist
    
    @property
    def get_models(self):
        return self.__models
    
    @property
    def get_models_config(self):
        return self.__models_conf
    
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
        self.__models.append(model)
        last_idx = len(self.__models)
        conf = self.__create_template_cofig__(model.get_name, model.get_weights, model.get_bias, last_idx, hasforward)
        self.__models_conf.append(conf)
        
    def run_forward(self, x):
        out = x
        for model in self.__models:
            out = model.forward(out)
        return out
    
    def calculate_delta(self, y_pred, y_true):
        delta = y_pred - y_true
        return delta
    
    def run_backprop_and_update(self, y_pred, y_true, lr=0.01):

        delta = self.calculate_delta(y_pred, y_true)

        for model in reversed(self.__models):
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

            self.__loss_hist.append(loss)
            self.run_backprop_and_update(y_pred, y_true, lr)
            
    def predict(self, x):
        return self.run_forward(x)