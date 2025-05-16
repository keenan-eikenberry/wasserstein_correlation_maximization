from torch import nn
from typing import Sequence, Optional, Union, Tuple
from wasserstein_correlation.models.conv_enums import PoolLayer

class CNN(nn.Module):
    """
    Convolutional neural network with flexible architecture specification.

    Args:
        conv_features (Sequence[Union[Tuple[int, int], PoolLayer]]): Convolutional features specified as tuples (in_channels, out_channels) and pooling layers (PoolLayer.Avg or PoolLayer.Max).
    """
    def __init__(self, 
                 input_shape: Tuple[int, ...], 
                 output_shape: Tuple[int, ...],
                 conv_features: Sequence[Union[Tuple[int, int], PoolLayer]], 
                 dense_features: Optional[Sequence[int]] = None, 
                 conv_kernel_size: int = 3, 
                 conv_stride: int = 1, 
                 conv_padding: int = 1, 
                 pool_kernel_size: int = 2, 
                 pool_stride: int = 2, 
                 pool_padding: int = 0, 
                 activation: str = 'leaky_relu',
                 leaky_param: float = 0.2, 
                 use_batch_norm: bool = True, 
                 use_dropout: bool = False, 
                 dropout_rate: float = 0.5):
        super(CNN, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.conv_features = conv_features
        self.dense_features = dense_features
        
        # Set up activation function
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'leaky_relu':
            self.activation = lambda: nn.LeakyReLU(leaky_param)
        elif activation == 'gelu':
            self.activation = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'relu', 'leaky_relu', or 'gelu'")
            
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for feature in self.conv_features:
            if isinstance(feature, tuple):
                in_channels, out_channels = feature
                self.conv_layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size, 
                    stride=conv_stride, 
                    padding=conv_padding
                ))
                if use_batch_norm:
                    self.conv_layers.append(nn.BatchNorm2d(num_features=out_channels))
                self.conv_layers.append(self.activation())
            elif isinstance(feature, PoolLayer):
                if feature == PoolLayer.Avg:
                    self.conv_layers.append(nn.AvgPool2d(
                        kernel_size=pool_kernel_size, 
                        stride=pool_stride, 
                        padding=pool_padding
                    ))
                elif feature == PoolLayer.Max:
                    self.conv_layers.append(nn.MaxPool2d(
                        kernel_size=pool_kernel_size, 
                        stride=pool_stride, 
                        padding=pool_padding
                    ))
    
        # Dense layers 
        if dense_features is not None:
            self.dense_layers = nn.ModuleList()
            for i in range(len(dense_features) - 1):
                self.dense_layers.append(nn.Linear(dense_features[i], dense_features[i + 1]))
                if i < len(dense_features) - 2:
                    self.dense_layers.append(self.activation())
                    if use_dropout:
                        self.dense_layers.append(nn.Dropout(p=dropout_rate))
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        if self.dense_features is not None:
            x = x.flatten(1)
            for layer in self.dense_layers:
                x = layer(x)
        
        return x