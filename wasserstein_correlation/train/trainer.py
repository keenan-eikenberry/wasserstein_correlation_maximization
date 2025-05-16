import torch
from torch import nn
import numpy as np
from torch.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
import os
import gc 
import logging
import shutil
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
import matplotlib.pyplot as plt
from wasserstein_correlation.config.train_config import TrainConfig
from wasserstein_correlation.config.model_config import ModelConfig
from wasserstein_correlation.train.train_factory import TrainFactory
from wasserstein_correlation.utils.yaml_utils import *
from wasserstein_correlation.utils.evaluate_structure import EvaluateStructure
from wasserstein_correlation.utils.evaluate_invariance import EvaluateInvariance 
from wasserstein_correlation.config.dataloader_config import DataloaderConfig

class Trainer:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        training_components = TrainFactory.create_training_components(self.train_config)

        self.run_config = train_config.run_config
        self.batch_size = self.run_config.batch_size
        self.epochs = self.run_config.epochs
        self.trainloader = training_components.trainloader
        self.testloader = training_components.testloader
        self.features = training_components.features
        self.encoder = training_components.encoder
        self.decoder = training_components.decoder
        self.augmentations = training_components.augmentations 
        self.num_aug_samples = self.run_config.num_aug_samples
        self.log_product_loss = self.run_config.log_product_loss
        self.loss_fn = training_components.loss_fn
        self.optimizer = training_components.optimizer
        self.scheduler = training_components.scheduler
        self.invariance_config = training_components.invariance_config

        # Initialize mixed precision training
        if self.run_config.use_autocast:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Set device
        self._validate_and_set_device()
        
        # Enable gradient checkpointing for encoder
        if hasattr(self.encoder, 'stages'): 
            self.encoder.use_checkpointing = True
        
        # Directory setup and logging initialization
        self._set_root_directory(self.train_config.run_config.root_dir) 
        self.save_dir_prefix = self.run_config.save_dir_prefix
        self.load_dir = self.run_config.load_dir
        self.start_epoch = 0

        if self.load_dir is not None:
            self._use_existing_directories()
            self._configure_logging(self.log_file, append=True)
            if self.run_config.checkpoint_filename is not None: 
                self._load_checkpoint(self.run_config.checkpoint_filename)
            else:
                self._load_checkpoint()
        else:
            self._create_save_directories()
            self._configure_logging(self.log_file, append=False)

        self._save_config_file()

        # Move models to device
        if self.features is not None:
            self.features.to(self.device)
        self.encoder.to(self.device)
        if self.decoder is not None:
            self.decoder.to(self.device)

    def _forward_pass(self, data):
        """Forward pass with optional gradient checkpointing"""
        if hasattr(self.encoder, 'use_checkpointing') and self.encoder.use_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            if self.encoder.training: 
                encoded = checkpoint.checkpoint(
                    create_custom_forward(self.encoder),
                    data,
                    use_reentrant=False 
                )
            else:
                encoded = self.encoder(data)
        else:
            encoded = self.encoder(data)
        return encoded

    def train(self):
        torch.cuda.empty_cache()
        gc.collect()

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.encoder.train()
            if self.decoder is not None:
                self.decoder.train()

            total_losses = []
            loss_terms_accumulator = {}

            with tqdm(total=len(self.trainloader), desc=f"Epoch {epoch + 1}/{self.start_epoch + self.epochs}") as pbar:
                for batch_idx, batch in enumerate(self.trainloader):
                    self.optimizer.zero_grad(set_to_none=self.run_config.grad_set_to_none)
                    self._maybe_clear_cache(batch_idx)
                    
                    if isinstance(batch, (tuple, list)):
                        data = batch[0].to(self.device)
                    else:
                        data = batch.to(self.device)
                    
                    if self.features is not None:
                        with torch.no_grad(): 
                            features = self.features(data)
                    
                    cm = autocast('cuda') if self.run_config.use_autocast else nullcontext()
                    with cm:
                        if self.features is not None: 
                            encoded = self._forward_pass(features)
                            input_distribution = features
                        else: 
                            encoded = self._forward_pass(data)
                            input_distribution = data 

                        if self.augmentations is not None:
                            augmented_encoded = self._apply_augmentations(data) 
                        else: 
                            augmented_encoded = None 
                    
                        if self.decoder is not None:
                            decoded = self.decoder(encoded)
                        else:
                            decoded = None
                        
                        # Compute loss
                        total_loss, loss_terms = self.loss_fn(
                            input_distribution, 
                            encoded, 
                            augmented_encoded,
                            decoded, 
                            self.run_config
                        )

                    # Backward pass
                    if self.run_config.use_autocast:
                        self.scaler.scale(total_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        total_loss.backward()
                        self.optimizer.step()
                    
                    self._maybe_step_scheduler(batch_idx=(batch_idx+1))
                    
                    # Update tracking
                    total_losses.append(total_loss.item())
                    for loss_name, loss_value in loss_terms.items():
                        if loss_name not in loss_terms_accumulator:
                            loss_terms_accumulator[loss_name] = []
                        loss_terms_accumulator[loss_name].append(loss_value)

                    pbar.set_postfix(loss=total_loss.item())
                    pbar.update(1)

                    if self.run_config.print_memory_and_optimizer_stats and batch_idx % 10 == 0:
                        self._print_memory_and_optimizer_stats()

            # Scheduler step
            if self.scheduler is not None and self.run_config.scheduler_batch_update is None:
                self.scheduler.step()  

            # Compute mean values
            with torch.no_grad():
                mean_total_loss = torch.tensor(total_losses).mean().item()
                mean_loss_terms = {
                    loss_name: torch.tensor(loss_value).mean().item()
                    for loss_name, loss_value in loss_terms_accumulator.items()
                }

            # Logging
            print(f"Epoch [{epoch + 1}/{self.start_epoch + self.epochs}] | Mean Total Loss: {mean_total_loss:.4f}")
            logging.info(f"Epoch [{epoch + 1}/{self.start_epoch + self.epochs}], Mean Total Loss: {mean_total_loss:.4f}")

            for loss_name, loss_value in mean_loss_terms.items():
                print(f"Epoch [{epoch + 1}/{self.start_epoch + self.epochs}] | Mean {loss_name}: {loss_value:.4f}")
                logging.info(f"Epoch [{epoch + 1}/{self.start_epoch + self.epochs}], Mean {loss_name}: {loss_value:.4f}")

            # Checkpoint saving
            if (epoch + 1) % self.run_config.checkpoint_epoch == 0:
                self._save_checkpoint(epoch)

            # Evaluation/visualization 
            if self.run_config.structure_tests is not None and (epoch + 1) % self.run_config.evaluate_structure_epoch == 0:
                self._evaluate_structure(epoch + 1)

            if self.run_config.invariance_tests is not None and (epoch + 1) % self.run_config.evaluate_invariance_epoch == 0:
                self._evaluate_invariance(epoch + 1)

            gc.collect()                
            self._maybe_clear_cache(epoch=epoch)

    def _apply_augmentations(self, data):
        augmented_encoded = []
        augmentations = list(self.augmentations.values())

        for aug in augmentations:
            collect_aug_encodings = []
            for _ in range(self.num_aug_samples):
                aug_data = aug(data)
                if self.features is not None: 
                    aug_features = self.features(aug_data)
                    aug_encoded = self._forward_pass(aug_features)
                else: 
                    aug_encoded = self._forward_pass(aug_data)
                collect_aug_encodings.append(aug_encoded)
            augmented_encoded.append(torch.cat(collect_aug_encodings, dim=0))

        if not self.log_product_loss:
            augmented_encoded = torch.cat(augmented_encoded, dim=0) 
        else: 
            augmented_encoded = torch.stack(augmented_encoded, dim=1)

        return augmented_encoded

    def _evaluate_structure(self, evaluate_epoch):
        plt.close('all')
        gc.collect()
        
        evaluate_structure = EvaluateStructure(
            features=self.features,
            encoder=self.encoder, 
            testloader=self.testloader,
            total_samples=None, 
            num_samples_per_class=200, 
            #Persistence analysis parameters
            top_max_dim=2,
            top_thresh=None,
            top_coeff=2,
            #Spectral analysis parameters
            spectral_k_values=[10, 20, 50], 
            num_eigvals=50,
            #Heat kernel analysis parameters
            heat_k_values=[10, 20, 50],
            diffusion_times=np.logspace(-1, 1, 10),
            #t-SNE parameters
            total_samples_tsne=None,
            num_samples_per_class_tsne=None,
            run_input_tsne=False, 
            tsne_perplexity=30,
            tsne_learning_rate=200,
            tsne_max_iter=2000, 
            tsne_metric='euclidean',
            data_filters=None,
            seed=self.run_config.seed, 
            evaluate_dir=self.evaluate_dir, 
            experiment_name=self.experiment_name
        )
        evaluate_structure.run_tests(
            tests=self.run_config.structure_tests, 
            epoch=evaluate_epoch
        )
        
        # Cleanup
        del evaluate_structure
        plt.close('all')
        gc.collect()

    def _evaluate_invariance(self, evaluate_epoch):
        plt.close('all')
        gc.collect()
        
        evaluate_invariance = EvaluateInvariance(
            device=self.device,
            num_classes=self.invariance_config.num_classes,
            trainloader=self.invariance_config.invariance_trainloader,
            testloader=self.invariance_config.invariance_testloader,
            features=self.features,
            encoder=self.encoder,
            augmentations=self.augmentations if self.invariance_config.augmentations is None else self.invariance_config.augmentations,
            # Classifier settings
            classifier_epochs=self.invariance_config.classifier_epochs,
            linear_classifier=self.invariance_config.linear_classifier_config,
            nonlinear_classifier=self.invariance_config.nonlinear_classifier_config,
            classifier_optimizer=self.invariance_config.classifier_optimizer_config,
            classifier_scheduler=self.invariance_config.classifier_scheduler_config,
            # End-to-end classifier settings
            end_to_end_classifier_epochs=self.invariance_config.end_to_end_classifier_epochs,
            end_to_end_classifier=self.invariance_config.end_to_end_classifier_config,
            end_to_end_optimizer=self.invariance_config.end_to_end_optimizer_config,
            end_to_end_scheduler=self.invariance_config.end_to_end_scheduler_config,
            # Additional parameters
            num_classification_runs=self.invariance_config.num_classification_runs,
            num_samples_per_class=self.invariance_config.num_samples_per_class,
            num_aug_samples=self.invariance_config.num_aug_samples,
            seed=self.run_config.seed,
            save_dir=self.save_dir)

        evaluate_invariance.run_evaluation(
            tests=self.run_config.invariance_tests, 
            epoch=evaluate_epoch
        ) 

        # Cleanup
        del evaluate_invariance
        plt.close('all')
        gc.collect()
    

    def _set_root_directory(self, root_dir): 
        current_dir = os.getcwd()

        if root_dir is not None: 
            if current_dir != root_dir:
                os.chdir(root_dir)
                print(f"Changed working directory to: {root_dir}")
            else:
                print(f"Current working directory is already: {root_dir}")
            self.root_dir = root_dir
        else: 
            self.root_dir = current_dir


    def _create_save_directories(self):
        """
        Creates a new experiment directory under `root_dir` with the provided prefix.
        Folders are numbered incrementally based on existing ones.
        
        Args:
            root_dir (str): The root directory where the `results` folder exists.
            prefix (str): A string prefix for the experiment name (e.g., 'STL10_invariance_test').
        """
        results_dir = os.path.join(self.root_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Extract all numbers from existing directories with the given prefix
        experiment_dirs = [d for d in os.listdir(results_dir) if d.startswith(self.save_dir_prefix)]
        existing_numbers = []
        
        for d in experiment_dirs:
            try:
                # Extract the number from the directory name
                parts = d.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    existing_numbers.append(int(parts[-1]))
            except (ValueError, IndexError):
                continue
        
        experiment_number = 1 if not existing_numbers else max(existing_numbers) + 1
        
        while True:
            self.experiment_name = f"{self.save_dir_prefix}_{experiment_number:03d}"
            self.save_dir = os.path.join(results_dir, self.experiment_name)
            
            if not os.path.exists(self.save_dir):
                break
            experiment_number += 1
        
        # Create subdirectories 
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        self.evaluate_dir = os.path.join(self.save_dir, 'evaluate')
        self.log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.evaluate_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)  

        self.log_file = os.path.join(self.log_dir, 'training.log')
        
        print(f"Created experiment directory: {self.save_dir}")


    def _use_existing_directories(self):
        self.save_dir = self.load_dir
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        self.evaluate_dir = os.path.join(self.save_dir, 'evaluate')
        self.log_dir = os.path.join(self.save_dir, 'logs')
        self.log_file = os.path.join(self.log_dir, 'training.log')
        self.experiment_name = os.path.basename(self.save_dir)

        print(f"Resuming training from: {self.save_dir}")
    

    def _save_config_file(self):
        config_dict = OrderedDict()

        config_dict['run_config'] = config_to_dict(self.train_config.run_config)
        config_dict['trainloader_config'] = config_to_dict(self.train_config.trainloader_config.to_dict())
        config_dict['testloader_config'] = config_to_dict(self.train_config.testloader_config.to_dict())

        if self.load_dir is not None:
            total_epochs = self.start_epoch + self.run_config.epochs
            config_dict['run_config']['epochs'] = total_epochs

        if self.features is not None and isinstance(self.train_config.features_config, ModelConfig): 
            config_dict['features_config'] = OrderedDict({
                'model_class': config_to_dict(self.train_config.features_config.model_class),
                'params': config_to_dict(self.train_config.features_config.params), 
                'custom_initialize': self.train_config.features_config.custom_initialize
            })
        elif self.features is not None and isinstance(self.train_config.features_config, nn.Module):
            config_dict['features_config'] = str(self.features.__class__.__name__)
        else: 
            config_dict['features_config'] = 'No features network'

        if isinstance(self.train_config.encoder_config, ModelConfig):
            config_dict['encoder_config'] = OrderedDict({
                'model_class': config_to_dict(self.train_config.encoder_config.model_class),
                'params': config_to_dict(self.train_config.encoder_config.params), 
                'custom_initialize': self.train_config.encoder_config.custom_initialize
            })
        elif isinstance(self.train_config.encoder_config, nn.Module):
            config_dict['encoder_config'] = str(self.train_config.encoder_config.__class__.__name__)

        if self.decoder is not None and isinstance(self.train_config.decoder_config, ModelConfig): 
            config_dict['decoder_config'] = OrderedDict({
                'model_class': config_to_dict(self.train_config.decoder_config.model_class),
                'params': config_to_dict(self.train_config.decoder_config.params), 
                'custom_initialize': self.train_config.decoder_config.custom_initialize
            })
        elif self.decoder is not None and isinstance(self.train_config.decoder_config, nn.Module):
            config_dict['decoder_config'] = str(self.train_config.decoder_config.__class__.__name__)
        else: 
            config_dict['decoder_config'] = 'No decoder'
        
        if self.train_config.augmentations is not None:
            config_dict['augmentations'] = OrderedDict(
                (k, str(v)) for k, v in self.train_config.augmentations.items()
            )
        else: 
            config_dict['augmentations'] = 'No augmentations'

        config_dict['loss_config'] = config_to_dict(self.train_config.loss_config)

        config_dict['optimizer_config'] = OrderedDict({
            'optimizer_class': f"optim.{self.train_config.optimizer_config.optimizer_class.__name__}",
            'params': config_to_dict(self.train_config.optimizer_config.params)
        })

        if self.scheduler is not None:
            config_dict['scheduler_config'] = OrderedDict({
                'scheduler_class': f"lr_scheduler.{self.train_config.scheduler_config.scheduler_class.__name__}",
                'params': config_to_dict(self.train_config.scheduler_config.params)
            })
        else: 
            config_dict['scheduler_config'] = 'No scheduler'

        if self.invariance_config is not None:
            invariance = OrderedDict()
            
            invariance['num_classes'] = self.invariance_config.num_classes

            if isinstance(self.train_config.invariance_config.invariance_trainloader, DataloaderConfig):
                invariance['invariance_trainloader'] = config_to_dict(self.train_config.invariance_config.invariance_trainloader.to_dict())
        
            if isinstance(self.train_config.invariance_config.invariance_trainloader, DataLoader):
                invariance['invariance_trainloader'] = OrderedDict({
                    'batch_size': self.train_config.invariance_config.invariance_trainloader.batch_size
                })

            if isinstance(self.train_config.invariance_config.invariance_testloader, DataloaderConfig):
                invariance['invariance_testloader'] = config_to_dict(self.train_config.invariance_config.invariance_testloader.to_dict())
        
            if isinstance(self.train_config.invariance_config.invariance_trainloader, DataLoader):
                invariance['invariance_testloader'] = OrderedDict({
                    'batch_size': self.train_config.invariance_config.invariance_trainloader.batch_size
                })
            
            invariance['classifier_epochs'] = self.invariance_config.classifier_epochs

            invariance['linear_classifier_config'] = OrderedDict({
                'model_class': config_to_dict(self.invariance_config.linear_classifier_config.model_class),
                'params': config_to_dict(self.invariance_config.linear_classifier_config.params),
                'custom_initialize': self.invariance_config.linear_classifier_config.custom_initialize
            })
            
            invariance['nonlinear_classifier_config'] = OrderedDict({
                'model_class': config_to_dict(self.invariance_config.nonlinear_classifier_config.model_class),
                'params': config_to_dict(self.invariance_config.nonlinear_classifier_config.params),
                'custom_initialize': self.invariance_config.nonlinear_classifier_config.custom_initialize
            })
            
            invariance['classifier_optimizer_config'] = OrderedDict({
                'optimizer_class': f"optim.{self.invariance_config.classifier_optimizer_config.optimizer_class.__name__}",
                'params': config_to_dict(self.invariance_config.classifier_optimizer_config.params)
            })
            
            invariance['classifier_scheduler_config'] = OrderedDict({
                'scheduler_class': f"lr_scheduler.{self.invariance_config.classifier_scheduler_config.scheduler_class.__name__}",
                'params': config_to_dict(self.invariance_config.classifier_scheduler_config.params)
            })
            
            invariance['end_to_end_classifier_epochs'] = self.invariance_config.end_to_end_classifier_epochs 
            
            if self.invariance_config.end_to_end_classifier_config is not None:
                invariance['end_to_end_classifier_config'] = OrderedDict({
                    'model_class': config_to_dict(self.invariance_config.end_to_end_classifier_config.model_class),
                    'params': config_to_dict(self.invariance_config.end_to_end_classifier_config.params),
                    'custom_initialize': self.invariance_config.end_to_end_classifier_config.custom_initialize
                })
            
            if self.invariance_config.end_to_end_optimizer_config is not None:
                invariance['end_to_end_optimizer_config'] = OrderedDict({
                    'optimizer_class': f"optim.{self.invariance_config.end_to_end_optimizer_config.optimizer_class.__name__}",
                    'params': config_to_dict(self.invariance_config.end_to_end_optimizer_config.params)
                })
            
            if self.invariance_config.end_to_end_scheduler_config is not None:
                invariance['end_to_end_scheduler_config'] = OrderedDict({
                    'scheduler_class': f"lr_scheduler.{self.invariance_config.end_to_end_scheduler_config.scheduler_class.__name__}",
                    'params': config_to_dict(self.invariance_config.end_to_end_scheduler_config.params)
                })
            
            invariance['num_classification_runs'] = self.invariance_config.num_classification_runs
            invariance['num_samples_per_class'] = self.invariance_config.num_samples_per_class
            invariance['num_aug_samples'] = self.invariance_config.num_aug_samples
            
            config_dict['invariance_config'] = invariance
        else:
            config_dict['invariance_config'] = 'No invariance tests'

        config_file_path = os.path.join(self.save_dir, "config.yaml")
        
        save_yaml_config(config_dict, config_file_path)

        print(f"Saved config file to {config_file_path}")

        
    def _configure_logging(self, log_file: str, append: bool=False):
        file_mode = 'a' if append else 'w'
        
        # Get the root logger
        logger = logging.getLogger()
        
        # Reset logger 
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Set level
        logger.setLevel(logging.INFO)
        
        # Create and add file handler
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logging.info(f"Logging configured to file: {log_file} (mode: {file_mode})")
    

    def _save_checkpoint(self, epoch):
        """
        Save the current state of the model, optimizer, and scheduler, etc. 
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.pth")

        checkpoint = {
        'epoch': epoch,
        'encoder': self.encoder.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'trainloader_rng_state': self.trainloader.generator.get_state()
        }
        
        if self.decoder is not None:
            checkpoint['decoder'] = self.decoder.state_dict()
        
        if self.scheduler is not None: 
            checkpoint['scheduler'] = self.scheduler.state_dict()

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Latest checkpoint
        latest_checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        shutil.copyfile(checkpoint_path, latest_checkpoint_path) 
        logging.info(f"Checkpoint saved at epoch {epoch + 1}")
      

    def _load_checkpoint(self, checkpoint_filename='latest_checkpoint.pth'):
        """
        Load the model, optimizer, scheduler, and epoch information from a checkpoint.
        """
        checkpoint_path = os.path.join(self.load_dir, 'checkpoints', checkpoint_filename)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=True)

            self.encoder.load_state_dict(checkpoint['encoder'])
            if 'decoder' in checkpoint and self.decoder is not None:
                self.decoder.load_state_dict(checkpoint['decoder'])

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            if self.scheduler and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1 

            # Restore RNG state
            torch.set_rng_state(checkpoint['rng_state'])
            if self.device.type == 'cuda' and checkpoint['cuda_rng_state'] is not None: 
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])            
            self.trainloader.generator.set_state(checkpoint['trainloader_rng_state'])

            print(f"Checkpoint loaded from: {checkpoint_path}. Resuming from epoch {self.start_epoch}")
            logging.info(f"Checkpoint loaded from: {checkpoint_path}. Resuming from epoch {self.start_epoch}")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Training from scratch.")
            logging.info(f"No checkpoint found at {checkpoint_path}. Training from scratch.")
    

    def _validate_and_set_device(self):
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
            return

        requested_device = self.run_config.set_gpu
        available_gpus = list(range(torch.cuda.device_count()))
        
        # String inputs
        if isinstance(requested_device, str):
            if requested_device == 'cuda':
                self.device = torch.device('cuda')
            elif requested_device.startswith('cuda:'):
                gpu_idx = int(requested_device.split(':')[1])
                if gpu_idx in available_gpus:
                    self.device = torch.device(requested_device)
                else:
                    raise ValueError(f"GPU {gpu_idx} not available. Available GPUs: {available_gpus}")
            else:
                raise ValueError(f"Invalid device string. Must be 'cuda' or 'cuda:N'. Available GPUs: {available_gpus}")
        # Integer inputs 
        elif isinstance(requested_device, int):
            if requested_device in available_gpus:
                self.device = torch.device(f'cuda:{requested_device}')
            else:
                raise ValueError(f"GPU {requested_device} not available. Available GPUs: {available_gpus}")
        else:
            raise ValueError(f"Invalid device specification. Must be string or integer. Available GPUs: {available_gpus}")


    def _maybe_step_scheduler(self, batch_idx):
        update_freq = self.run_config.scheduler_batch_update
        if self.scheduler and update_freq and update_freq % batch_idx == 0:
            self.scheduler.step()


    def _maybe_clear_cache(self, batch_idx=None, epoch=None):
        if not torch.cuda.is_available():
            return
            
        if batch_idx is not None:
            # Batch-level cache clearing
            if self.run_config.clear_cache_batch and batch_idx % self.run_config.clear_cache_batch == 0:
                torch.cuda.empty_cache()
        elif epoch is not None:
            # Epoch-level cache clearing
            if self.run_config.clear_cache_epoch and epoch % self.run_config.clear_cache_epoch == 0:
                torch.cuda.empty_cache()
    
    
    def _print_memory_and_optimizer_stats(self):
        print("\n\nMEMORY STATS:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        for i in range(torch.cuda.device_count()):
            print(f"Memory reserved on GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

        print("\nOPTIMIZER STATS:")
        for i, group in enumerate(self.optimizer.param_groups):
            total_state_size = sum(
                sum(s.numel() for s in state.values() if torch.is_tensor(s))
                for state in self.optimizer.state.values()
            )
            print(f"Parameter group {i} state size: {total_state_size}\n")