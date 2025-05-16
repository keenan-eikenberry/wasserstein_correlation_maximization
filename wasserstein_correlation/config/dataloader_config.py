from typing import Union
from torch.utils.data import Dataset

class DataloaderConfig:
    def __init__(self, 
                 dataset: Union[Dataset, str],
                 params: dict):
                 
        self.dataset = dataset
        self.params = params

    def to_dict(self):
        if isinstance(self.dataset, str):
            return {
                'dataset': str(self.dataset),
                'params': self.params
            }
        else: 
            return {
                'dataset': {
                    'root': getattr(self.dataset, 'root', 'unknown_path'),
                    'train': getattr(self.dataset, 'train', 'unknown'),
                    'transform': str(self.dataset.transform) if hasattr(self.dataset, 'transform') else 'None',
                    'target_transform': str(self.dataset.target_transform) if hasattr(self.dataset, 'target_transform') else 'None'
                },
                'params': self.params
            }