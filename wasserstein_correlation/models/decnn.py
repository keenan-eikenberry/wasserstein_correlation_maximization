from torch import nn
from typing import Sequence, Optional, Tuple
from wasserstein_correlation.models.conv_enums import ConvLayer

class DeCNN(nn.Module):
    """
    Deconvolutional neural network with flexible architecture specification.

    Args:
        conv_features (Sequence[Tuple[ConvLayer, int, int]]): Sequence of tuples specifying:
            (layer_type, in_channels, out_channels), where layer_type is Conv, Tran, or TranZero.
    """
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 output_shape: Tuple[int, int, int],
                 dense_features: Optional[Sequence[int]], 
                 conv_input_shape: Tuple[int, int, int], 
                 conv_features: Sequence[Tuple[ConvLayer, int, int]],  
                 conv_kernel_size: int = 3, 
                 conv_stride: int = 1, 
                 conv_padding: int = 1, 
                 transpose_kernel_size: int = 3, 
                 transpose_stride: int = 2, 
                 transpose_padding: int = 1, 
                 transpose_output_padding: int = 1, 
                 output_kernel_size: int = 3, 
                 output_stride: int = 1, 
                 output_padding: int = 1,
                 activation: str = 'leaky_relu',
                 leaky_param: float = 0.2,
                 use_batch_norm: bool = True, 
                 use_dropout: bool = False, 
                 dropout_rate: float = 0.5, 
                 output_activation: Optional[str] = None):
        super(DeCNN, self).__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.conv_input_shape = conv_input_shape
        
        # Set up activation function
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'leaky_relu':
            self.activation = lambda: nn.LeakyReLU(leaky_param)
        elif activation == 'gelu':
            self.activation = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'relu', 'leaky_relu', or 'gelu'")
        
        # Dense layers 
        if dense_features is not None:
            self.dense_layers = nn.ModuleList()
            for i in range(len(dense_features) - 1):
                self.dense_layers.append(nn.Linear(dense_features[i], dense_features[i + 1]))
                if i < len(dense_features) - 2:
                    self.dense_layers.append(self.activation())
                    if use_dropout:
                        self.dense_layers.append(nn.Dropout(p=dropout_rate))
        else:
            self.dense_layers = None

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for layer_type, in_channels, out_channels in conv_features:
            if layer_type == ConvLayer.Conv:
                self.conv_layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=conv_padding
                ))
            elif layer_type == ConvLayer.Tran:
                self.conv_layers.append(nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=transpose_kernel_size,
                    stride=transpose_stride,
                    padding=transpose_padding,
                    output_padding=transpose_output_padding
                ))
            elif layer_type == ConvLayer.TranZero:
                self.conv_layers.append(nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=transpose_kernel_size,
                    stride=transpose_stride,
                    padding=0,
                    output_padding=0
                ))
                
            if use_batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(num_features=out_channels))
            self.conv_layers.append(self.activation())
        
        # Output layer
        self.output_layer = nn.ModuleList()
        self.output_layer.append(nn.ConvTranspose2d(
            in_channels=conv_features[-1][2], 
            out_channels=output_shape[0], 
            kernel_size=output_kernel_size, 
            stride=output_stride, 
            padding=output_padding
        ))
        
        # Optional output activation
        if output_activation == 'sigmoid':
            self.output_layer.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            self.output_layer.append(nn.Tanh())
    
    def forward(self, x):
        if self.dense_layers is not None:
            if x.dim() > 2:
                x = x.flatten(1)
            for layer in self.dense_layers:
                x = layer(x)
        
        x = x.reshape(-1, *self.conv_input_shape)
        
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Pass through output layer
        for layer in self.output_layer:
            x = layer(x)

        return x