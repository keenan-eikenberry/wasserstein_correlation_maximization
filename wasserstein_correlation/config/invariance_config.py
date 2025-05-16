from torch.utils.data import DataLoader
from typing import Optional, Union, Dict, Callable
from wasserstein_correlation.config.model_config import ModelConfig
from wasserstein_correlation.config.optimizer_config import OptimizerConfig
from wasserstein_correlation.config.scheduler_config import SchedulerConfig
from wasserstein_correlation.config.dataloader_config import DataloaderConfig


class InvarianceConfig: 
     def __init__(self, 
                  num_classes: int,
                  augmentations: Optional[Dict[str, Callable]],
                  invariance_trainloader: Union[DataLoader, DataloaderConfig],
                  invariance_testloader: Union[DataLoader, DataloaderConfig],
                  # Classifier 
                  classifier_epochs: int,
                  linear_classifier_config: ModelConfig,
                  nonlinear_classifier_config: ModelConfig,
                  classifier_optimizer_config: OptimizerConfig,
                  classifier_scheduler_config: SchedulerConfig,
                  # End to end classifier
                  end_to_end_classifier_epochs: Optional[int]=None, 
                  end_to_end_classifier_config: Optional[ModelConfig]=None,
                  end_to_end_optimizer_config: Optional[OptimizerConfig]=None,
                  end_to_end_scheduler_config: Optional[SchedulerConfig]=None, 
                  # Additional parameters 
                  num_classification_runs: int=5, 
                  num_samples_per_class: Optional[int]=None,
                  num_aug_samples: int = 200):
        
        self.num_classes = num_classes
        self.augmentations = augmentations 
        self.invariance_trainloader = invariance_trainloader
        self.invariance_testloader = invariance_testloader
        self.classifier_epochs = classifier_epochs
        self.linear_classifier_config = linear_classifier_config
        self.nonlinear_classifier_config = nonlinear_classifier_config
        self.classifier_optimizer_config = classifier_optimizer_config
        self.classifier_scheduler_config = classifier_scheduler_config
        self.end_to_end_classifier_epochs = end_to_end_classifier_epochs
        self.end_to_end_classifier_config = end_to_end_classifier_config
        self.end_to_end_optimizer_config = end_to_end_optimizer_config
        self.end_to_end_scheduler_config = end_to_end_scheduler_config
        self.num_classification_runs = num_classification_runs
        self.num_samples_per_class = num_samples_per_class 
        self.num_aug_samples = num_aug_samples