from torch import nn
import inspect
from wasserstein_correlation.config.model_config import ModelConfig
from wasserstein_correlation.models.model_classes import Model

class ModelFactory:
    """
    Create model from ModelConfig class
    """
    @staticmethod
    def create_model_config(model_config: ModelConfig):
        model_class = model_config.model_class
        model_params = model_config.params

        # Recursively call ModelFactory for any sub-networks passed as ModelConfig
        for param_name, param_value in model_params.items():
            if isinstance(param_value, ModelConfig):
                # Recursively create any sub-networks 
                model_params[param_name] = ModelFactory.create_model_config(param_value)

        # Instantiate the model
        model = ModelFactory._instantiate_model(
            model_class=model_class,
            model_params=model_params
        )

        # Initialization 
        if model_config.custom_initialize: 
            ModelFactory._custom_initialization(model)

        return model

    @staticmethod
    def _custom_initialization(model: nn.Module):
        # Optional custom initialiation
        def initialize_weights(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0)

            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight.data, 1.0)
                nn.init.constant_(layer.bias.data, 0.0)

            elif isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0)

            # Default initialization 
            elif hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        model.apply(initialize_weights)
  
    @staticmethod
    def _instantiate_model(model_class: Model, model_params: dict):
        """
        Initialize model using the filtered parameters matching model's constructor.
    
        Args:
            model_class (Model)
            model_params (dict)
        """
        model = model_class.value
        model_signature = inspect.signature(model)

        # Filter out non-constructor arguments 
        filtered_params = {k: v for k, v in model_params.items() if k in model_signature.parameters}

        # Instantiate with filtered parameters
        return model(**filtered_params)