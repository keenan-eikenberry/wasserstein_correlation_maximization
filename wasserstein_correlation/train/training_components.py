from torch import nn
from torch.utils.data import DataLoader
from typing import Callable, Optional, Dict
from wasserstein_correlation.config.run_config import RunConfig
from wasserstein_correlation.config.invariance_config import InvarianceConfig

class TrainingComponents:
    def __init__(self, 
                 run_config: RunConfig, 
                 trainloader: DataLoader,
                 testloader: DataLoader, 
                 features: Optional[nn.Module],
                 encoder: nn.Module,
                 decoder: Optional[nn.Module],
                 augmentations: Optional[Dict[str, Callable]],
                 loss_fn: Callable, 
                 optimizer: Callable, 
                 scheduler: Optional[Callable], 
                 invariance_config: Optional[InvarianceConfig]):
        
        self.run_config = run_config
        self.trainloader = trainloader
        self.testloader = testloader
        self.features = features
        self.encoder = encoder
        self.decoder = decoder
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.invariance_config = invariance_config