from modules.base_model import DeepLearningBaseModel
import os
import json
import numpy as np
from datetime import datetime

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
            hasforward: True | False,
            activation: ....
            pramnumber: ...
            size: ...
        }
        """
        self.__models_conf = []
        self.__models = []
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
    
    def __create_template_cofig__(
        self, name: str, idx: int, 
        w: np.ndarray, b: np.ndarray, 
        size: float, paramnumber: int,
        activation: str,
        hasforward: bool = True,
    ):
        return {
            "name": name,
            "params": {
                "weights": w,
                "bias": b,
            },
            "index": idx,
            "hasforward": hasforward,
            "size": size,
            "activation": activation,
            "pramnumber": paramnumber
        }
    
    def add(self, model:DeepLearningBaseModel, hasforward: bool=True):
        self.__models.append(model)
        last_idx = len(self.__models)
        
        model_weights, model_bias = model.get_params
        param_number = model.get_param_size
        param_mem_size = model.getparam_memory
        activation_function = model.activation_function_key
        hasforward = True if model.activation_function_key is not None else False
        
        conf = self.__create_template_cofig__(
            model.get_name, last_idx,
            model_weights, model_bias,
            param_number, param_mem_size,
            activation_function,
            hasforward
        )
        
        self.__models_conf.append(conf)

    def save_model_config(
        self,
        train_acc: float = 0.0,
        test_acc: float = 0.0,
        base_folder: str = "folder_results"
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{base_folder}_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)

        config_output = []

        for conf in self.__models_conf:
            name = conf["name"]
            idx = conf["index"]

            has_forward = conf["hasforward"]

            w_filename = None
            b_filename = None

            if has_forward:
                weights = conf["params"]["weights"]
                bias = conf["params"]["bias"]

                w_filename = f"{name}-{idx}-weights.npz"
                b_filename = f"{name}-{idx}-bias.npz"

                w_path = os.path.join(folder_name, w_filename)
                b_path = os.path.join(folder_name, b_filename)

                np.savez_compressed(w_path, weights=weights)
                np.savez_compressed(b_path, bias=bias)

            config_output.append({
                "name": f"{name}-{idx}",
                "wparam": w_filename,
                "bparam": b_filename,
                "activation": conf["activation"],
                "bytesize": str(conf["size"]),
                "forward": has_forward
            })

        result_json = {
            "timestamp": timestamp,
            "trainaccuracy": train_acc,
            "testaccuracy": test_acc,
            "configuration": config_output
        }

        json_path = os.path.join(folder_name, "model_results.json")
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=4)

        print(f"Model saved in folder: {folder_name}")