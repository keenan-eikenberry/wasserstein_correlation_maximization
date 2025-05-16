from enum import Enum

# Average vs. max pooling 
class PoolLayer(Enum):
    Avg = 'average pooling'
    Max = 'maximum pooling'


# Regular convolution vs. convolutional transpose
class ConvLayer(Enum): 
    Conv = 'convolution'
    Tran = 'transpose convolution'
    TranZero = 'transpose convolution with zero padding'