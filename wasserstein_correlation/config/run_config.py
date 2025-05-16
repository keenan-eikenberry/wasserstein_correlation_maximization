from typing import Optional, List

class RunConfig:
    def __init__(self, 
                 root_dir: str,
                 dataset: str,
                 invariance_dataset: str, 
                 save_dir_prefix: str,
                 batch_size: int,
                 epochs: int,
                 checkpoint_epoch: int=10,
                 evaluate_structure_epoch: Optional[int]=50,
                 structure_tests: Optional[List[str]]=None,
                 evaluate_invariance_epoch: Optional[int]=50,
                 invariance_tests: Optional[List[str]]=None,
                 num_wasserstein_projections: int=1000,
                 wasserstein_p: float=2.0,
                 wasserstein_q: float=2.0,
                 standardize_correlation: bool=False,
                 num_augs: Optional[int]=None,
                 num_aug_samples: Optional[int]=None, 
                 log_product_loss: Optional[bool]=False,
                 seed: Optional[int]=None,
                 load_dir: Optional[str]=None,
                 checkpoint_filename: Optional[str]=None,
                 set_gpu: Optional[str]=None,
                 scheduler_batch_update: Optional[int]=None,
                 use_autocast: bool=False,
                 grad_set_to_none: bool=False,
                 clear_cache_batch: Optional[int]=None,
                 clear_cache_epoch: Optional[int]=None,
                 print_memory_and_optimizer_stats: bool=False,
                 training_run_info: Optional[str]=None):
        
        self.root_dir = root_dir
        self.dataset = dataset
        self.invariance_dataset = invariance_dataset
        self.save_dir_prefix = save_dir_prefix
        self.batch_size = batch_size 
        self.epochs = epochs
        self.checkpoint_epoch = checkpoint_epoch
        self.evaluate_structure_epoch = evaluate_structure_epoch
        self.structure_tests = structure_tests
        self.evaluate_invariance_epoch = evaluate_invariance_epoch
        self.invariance_tests = invariance_tests
        self.num_wasserstein_projections = num_wasserstein_projections
        self.wasserstein_p = wasserstein_p
        self.wasserstein_q = wasserstein_q
        self.standardize_correlation = standardize_correlation
        self.num_augs = num_augs
        self.num_aug_samples = num_aug_samples
        self.log_product_loss = log_product_loss
        self.seed = seed
        self.load_dir = load_dir 
        self.checkpoint_filename = checkpoint_filename
        self.set_gpu = set_gpu
        self.scheduler_batch_update = scheduler_batch_update 
        self.use_autocast = use_autocast
        self.grad_set_to_none = grad_set_to_none
        self.clear_cache_batch = clear_cache_batch
        self.clear_cache_epoch = clear_cache_epoch
        self.print_memory_and_optimizer_stats = print_memory_and_optimizer_stats
        self.training_run_info = training_run_info