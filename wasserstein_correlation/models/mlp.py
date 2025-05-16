from torch import nn
from typing import Sequence, Tuple, Optional

class MLP(nn.Module): 
    def __init__(self, 
                 input_shape: Tuple[int, ...], 
                 output_shape: Tuple[int, ...],
                 dense_features: Sequence[int],
                 normalization_layer: Optional[str] = None, 
                 activation: str = 'relu',
                 use_dropout: bool = False, 
                 dropout_rate: float = 0.5,
                 output_activation: Optional[str] = None): 
        super(MLP, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dense_features = dense_features
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate 
        
        # Set activation function
        activation = activation.lower()
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'gelu':
            self.activation = nn.GELU
        else:
            raise ValueError(f"Unsupported activation function '{activation}'. Choose 'relu' or 'gelu'.")
        
        # Set optional normalization layer
        self.normalization = None
        if normalization_layer: 
            normalization_layer = normalization_layer.lower()
            if normalization_layer == 'batch_norm':
                self.normalization = nn.BatchNorm1d
            elif normalization_layer == 'layer_norm':
                self.normalization = nn.LayerNorm
            else: 
                raise ValueError(f"Unsupported normalization layer '{normalization_layer}'. Choose 'batch_norm' or 'layer_norm'.")
        
        # Set optional output activation function
        self.output_act_fn = None
        if output_activation:
            output_activation = output_activation.lower()
            if output_activation == 'sigmoid':
                self.output_act_fn = nn.Sigmoid
            elif output_activation == 'tanh':
                self.output_act_fn = nn.Tanh
            elif output_activation == 'softmax':
                self.output_act_fn = lambda: nn.Softmax(dim=1)
            else:
                raise ValueError(f"Unsupported output activation '{output_activation}'. Choose 'sigmoid', 'tanh' or 'softmax'.")

        # Dense layers
        self.dense_layers = nn.ModuleList()
        for i in range(len(dense_features) - 1):
            self.dense_layers.append(nn.Linear(dense_features[i], dense_features[i + 1]))
            
            if self.normalization:
                self.dense_layers.append(self.normalization(dense_features[i + 1]))
                
            if i < len(dense_features) - 2:
                self.dense_layers.append(self.activation())
                if self.use_dropout: 
                    self.dense_layers.append(nn.Dropout(p=self.dropout_rate))
        
        # Optional output activation
        if self.output_act_fn:
            self.dense_layers.append(self.output_act_fn())
          
    def forward(self, x): 
        if x.dim() > 2:
            x = x.flatten(1)
        for layer in self.dense_layers:
            x = layer(x)
        
        x = x.reshape(-1, *self.output_shape)
        
        return x