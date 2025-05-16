from torch import nn
from typing import Optional, Union, Callable, Dict 
from wasserstein_correlation.config.run_config import RunConfig
from wasserstein_correlation.config.dataloader_config import DataloaderConfig
from wasserstein_correlation.config.model_config import ModelConfig
from wasserstein_correlation.config.loss_config import LossConfig
from wasserstein_correlation.config.optimizer_config import OptimizerConfig 
from wasserstein_correlation.config.scheduler_config import SchedulerConfig
from wasserstein_correlation.config.invariance_config import InvarianceConfig

class TrainConfig:
    def __init__(self, 
                 run_config: RunConfig, 
                 trainloader_config: DataloaderConfig,
                 testloader_config: DataloaderConfig,
                 features_config: Optional[Union[nn.Module, ModelConfig]],
                 encoder_config: Union[nn.Module, ModelConfig],
                 decoder_config: Optional[Union[nn.Module, ModelConfig]],
                 augmentations: Optional[Dict[str, Callable]], 
                 loss_config: LossConfig, 
                 optimizer_config: OptimizerConfig, 
                 scheduler_config: Optional[SchedulerConfig], 
                 invariance_config: Optional[InvarianceConfig]):
        
        self.run_config = run_config
        self.trainloader_config = trainloader_config
        self.testloader_config = testloader_config 
        self.features_config = features_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.augmentations = augmentations
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.invariance_config = invariance_config