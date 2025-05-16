from wasserstein_correlation.models.model_classes import Model

class ModelConfig:
    """
    Model configuration.
    
    Args:
        model_class (Model): Specifies class of model (e.g., 'MLP', 'CNN', etc.).

        model_params (dict): Specifies model parameters.
        
        initialize (bool): Initialization flag. Defaults to False.
    """
    def __init__(self, 
                 model_class: Model, 
                 params: dict, 
                 custom_initialize: bool=False):

        self.model_class = model_class
        self.params = params
        self.custom_initialize = custom_initialize