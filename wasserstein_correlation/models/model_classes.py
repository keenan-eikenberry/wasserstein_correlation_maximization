from wasserstein_correlation.models.mlp import MLP
from wasserstein_correlation.models.cnn import CNN
from wasserstein_correlation.models.decnn import DeCNN
from wasserstein_correlation.models.patch_encoder import PatchEncoder
from wasserstein_correlation.models.patch_decoder import PatchDecoder
from enum import Enum

class Model(Enum):
    MLP = MLP
    CNN = CNN
    DeCNN = DeCNN
    PatchEncoder = PatchEncoder
    PatchDecoder = PatchDecoder