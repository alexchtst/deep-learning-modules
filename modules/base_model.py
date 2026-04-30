class DeepLearningBaseModel:
    def __init__(self, name, W=None, b=None, activation_function_key=None):
        self.name = name
        self.W = W
        self.b = b
        self.activation_function_key = activation_function_key
    
    @property
    def get_name(self):
        return self.name
    
    @property
    def get_params(self):
        return self.W, self.b
    
    @property
    def get_param_size(self):
        if self.b != None:
            return self.b.size + self.W.size
        return self.W.size
    
    @property
    def is_valid(self):
        if self.W == None:
            return False
        
        if self.name == None:
            return False