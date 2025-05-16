import os
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Optional, Union, Callable
import logging
from collections import defaultdict

from wasserstein_correlation.models.model_factory import ModelFactory
from wasserstein_correlation.utils.config_imports import * 
from wasserstein_correlation.utils.aux_functions import filter_parameters


class EncoderClassifierModel(nn.Module):
    """Wrapper that composes feature extractor, encoder, and classifier"""
    
    def __init__(self, 
                 features: Optional[nn.Module], 
                 encoder: Optional[nn.Module], 
                 classifier: nn.Module):
        super().__init__()
        self.features = features  
        self.encoder = encoder  
        self.classifier = classifier  
    
    def forward(self, x):
        if self.features is not None:
            features_out = self.features(x)
        else:
            features_out = x
        
        if self.encoder is not None: 
            encoder_out = self.encoder(features_out)
        else: 
            encoder_out = features_out
        
        return self.classifier(encoder_out)


class ClassifierTrainer:
    """Trainer for classifier models"""
    
    def __init__(self,
                 device: str, 
                 classifier: Union[ModelConfig, nn.Module],
                 optimizer_config: Optional[OptimizerConfig],
                 scheduler_config: Optional[SchedulerConfig],
                 epochs: int):
        self.device = device
        self.epochs = epochs
        
        # Create classifier model
        if isinstance(classifier, ModelConfig):
            self.classifier = ModelFactory.create_model_config(classifier)
        else:
            self.classifier = classifier 

        self.classifier.to(self.device)

        # Setup optimizer and scheduler
        if optimizer_config:
            optimizer_class = optimizer_config.optimizer_class
            valid_params = filter_parameters(optimizer_class, optimizer_config.params)
            
            self.optimizer = optimizer_class(self.classifier.parameters(), **valid_params)
        else:
            # Default optimizer
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
            
        if scheduler_config:
            scheduler_class = scheduler_config.scheduler_class
            valid_params = filter_parameters(scheduler_class, scheduler_config.params)

            self.scheduler = scheduler_class(self.optimizer, **valid_params)
        else:
            # Default to no scheduler 
            self.scheduler = None
        
    def train(self,
              trainloader, 
              features: Optional[nn.Module]=None, 
              encoder: Optional[nn.Module]=None):
    
        stats = {'loss': [], 'accuracy': []}
        
        if features is not None:
            features.eval()
        if encoder is not None:
            encoder.eval()
        
        for epoch in range(self.epochs):
            self.classifier.train()
            
            total_losses = []
            correct = 0
            total = 0
            
            with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}/{self.epochs}") as pbar:
                for batch_idx, batch in enumerate(trainloader):
                    self.optimizer.zero_grad()
                    
                    # Extract data and targets
                    data = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                   
                    # Forward through feature extractor and encoder
                    with torch.no_grad():
                        if features is not None:
                            features_out = features(data)
                        else:
                            features_out = data
                        
                        if encoder is not None:
                            encoder_out = encoder(features_out)
                        else:
                            encoder_out = features_out
                    
                    outputs = self.classifier(encoder_out)
                    
                    loss = F.cross_entropy(outputs, targets)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_losses.append(loss.item())
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    current_acc = 100.0 * correct / total
                    pbar.set_postfix(loss=loss.item(), accuracy=f"{current_acc:.2f}%")
                    pbar.update(1)
            
            if self.scheduler:
                self.scheduler.step()
            
            epoch_loss = torch.tensor(total_losses).mean().item()
            epoch_acc = 100.0 * correct / total
            stats['loss'].append(epoch_loss)
            stats['accuracy'].append(epoch_acc)
            
            print(f"Epoch [{epoch + 1}/{self.epochs}] | Mean Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
        # Create the complete model
        model = EncoderClassifierModel(features, encoder, self.classifier)
        
        return model, stats


class EvaluateInvariance:
    """
    1. Evaluate the classification performance of encoder on original vs augmented data  
    2. Optionally compare against end-to-end supervised classifier on data space/feature space
    3. Measure invariance directly by computing L2 and cosine similarity distances between original and augmented encodings
    """
    def __init__(self,
                 device: str,
                 num_classes: int, 
                 trainloader: DataLoader,
                 testloader: DataLoader,
                 features: Optional[nn.Module],
                 encoder: nn.Module,
                 augmentations: Dict[str, Callable],
                 # Classifier settings
                 classifier_epochs: int,
                 linear_classifier: Union[ModelConfig, nn.Module],
                 nonlinear_classifier: Union[ModelConfig, nn.Module],
                 classifier_optimizer: Optional[OptimizerConfig],
                 classifier_scheduler: Optional[SchedulerConfig],
                 # End-to-end classifier settings
                 end_to_end_classifier_epochs: Optional[int],
                 end_to_end_classifier: Optional[Union[ModelConfig, nn.Module]],
                 end_to_end_optimizer: Optional[OptimizerConfig],
                 end_to_end_scheduler: Optional[SchedulerConfig],
                 # Additional parameters
                 num_classification_runs: int=5, 
                 num_samples_per_class: Optional[int]=None,
                 num_aug_samples: int = 200,
                 seed: int = 42,
                 save_dir: str = 'default_save_dir'):
        
        self.device = device 
        self.num_classes = num_classes

        # Dataloaders
        self.trainloader = trainloader
        self.testloader = testloader
        
        # Setup features
        if features is not None: 
            self.features = features 
            self.features.to(self.device)
            self.features.eval() 
        else:
            self.features = None
        
        # Setup encoder
        self.encoder = encoder 
        self.encoder.to(self.device)
        self.encoder.eval()
            
        # Classifiers 
        self.classifier_epochs = classifier_epochs
        self.linear_classifier = linear_classifier
        self.nonlinear_classifier = nonlinear_classifier
        self.classifier_optimizer = classifier_optimizer
        self.classifier_scheduler = classifier_scheduler 
        
        self.end_to_end_classifier_epochs = end_to_end_classifier_epochs
        self.end_to_end_classifier = end_to_end_classifier
        self.end_to_end_optimizer = end_to_end_optimizer
        self.end_to_end_scheduler = end_to_end_scheduler 
        
        # Augmentations
        self.augmentations = augmentations 
        
        # Additional evaluation parameters
        self.num_classification_runs = num_classification_runs
        if num_samples_per_class is None: 
            self.num_samples_per_class = self._get_max_samples(testloader)
        else: 
            self.num_samples_per_class = num_samples_per_class
        self.num_aug_samples = num_aug_samples

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        # Save directory and logger
        self.save_dir = save_dir
        self._create_save_directories()
        self._configure_logging()
    

    def train_classifiers(self):
        self.logger.info("Training classifiers...")
    
        self.logger.info("Training classifiers for encoder...")
        linear_encoder_classifier_trainer = ClassifierTrainer(
            device=self.device,
            classifier=self.linear_classifier,
            optimizer_config=self.classifier_optimizer,
            scheduler_config=self.classifier_scheduler,
            epochs=self.classifier_epochs
        )
        self.linear_encoder_model, self.linear_encoder_classifier_stats = linear_encoder_classifier_trainer.train(
            trainloader=self.trainloader, 
            features=self.features, 
            encoder=self.encoder
        )

        nonlinear_encoder_classifier_trainer = ClassifierTrainer(
            device=self.device,
            classifier=self.nonlinear_classifier,
            optimizer_config=self.classifier_optimizer,
            scheduler_config=self.classifier_scheduler,
            epochs=self.classifier_epochs
        )
        self.nonlinear_encoder_model, self.nonlinear_encoder_classifier_stats = nonlinear_encoder_classifier_trainer.train(
            trainloader=self.trainloader, 
            features=self.features, 
            encoder=self.encoder
        )
        
        # Optionally train end-to-end classifier/feature space classifier
        if self.end_to_end_classifier:
            self.logger.info("Training end-to-end classifier/feature space classifier...")
            end_to_end_classifier_trainer = ClassifierTrainer(
                device=self.device,
                classifier=self.end_to_end_classifier,
                optimizer_config=self.end_to_end_optimizer,
                scheduler_config=self.end_to_end_scheduler,
                epochs=self.end_to_end_classifier_epochs
            )
            self.end_to_end_model, self.end_to_end_classifier_stats = end_to_end_classifier_trainer.train(
                    trainloader=self.trainloader, 
                    features=self.features,
                    encoder=None
                )
            
        self.logger.info("Classifier training complete")
    
    def evaluate_classification(self, 
                            augmentation: Optional[Callable] = None, 
                            aug_name: str = "original", 
                            epoch=None):
        """
        Evaluate classification performance on (possibly augmented) test data.
        If augmentation is provided, run multiple times and average results.
        """
        epoch_str = f'epoch_{epoch}_' if epoch is not None else ''

        if augmentation:
            desc = f"with {aug_name}"
            self.logger.info(f"Evaluating classification {desc} ({self.num_classification_runs} runs)...")
            
            accumulated_results = {}
            
            for run in range(self.num_classification_runs):
                self.logger.info(f"Run {run+1}/{self.num_classification_runs}...")
                
                run_results = {}
                # Encoder
                linear_encoder_results = self._evaluate_model_accuracy(
                    model=self.linear_encoder_model,
                    dataloader=self.testloader,
                    augmentation=augmentation,
                    model_name="Encoder w/ Linear Classifier",
                    aug_name=aug_name
                )
                run_results['linear_encoder'] = linear_encoder_results

                nonlinear_encoder_results = self._evaluate_model_accuracy(
                    model=self.nonlinear_encoder_model,
                    dataloader=self.testloader,
                    augmentation=augmentation,
                    model_name="Encoder w/ Nonlinear Classifier",
                    aug_name=aug_name
                )
                run_results['nonlinear_encoder'] = nonlinear_encoder_results
        
                # End-to-end classifier/feature space classifier 
                if self.end_to_end_classifier:
                    end_to_end_results = self._evaluate_model_accuracy(
                        model=self.end_to_end_model,
                        dataloader=self.testloader,
                        augmentation=augmentation,
                        model_name="End-to-End Classifier/Feature Space Classifier",
                        aug_name=aug_name
                    )
                    run_results['end_to_end'] = end_to_end_results
                
                # Initialize accumulated results
                if run == 0:
                    for model_key, model_results in run_results.items():
                        accumulated_results[model_key] = {
                            'accuracy_sum': model_results['accuracy'],
                            'class_accuracy_sum': {
                                cls: acc for cls, acc in model_results['class_accuracy'].items()
                            },
                            'runs': 1
                        }
                else:
                    for model_key, model_results in run_results.items():
                        accumulated_results[model_key]['accuracy_sum'] += model_results['accuracy']

                        for cls, acc in model_results['class_accuracy'].items():
                            accumulated_results[model_key]['class_accuracy_sum'][cls] += acc

                        accumulated_results[model_key]['runs'] += 1
                        
            # Average results across runs
            final_results = {}
            for model_key, acc_data in accumulated_results.items():
                runs = acc_data['runs']
                final_results[model_key] = {
                    'accuracy': acc_data['accuracy_sum'] / runs,
                    'class_accuracy': {
                        cls: acc_sum / runs 
                        for cls, acc_sum in acc_data['class_accuracy_sum'].items()
                    }
                }
                
                # Log averaged results
                model_name = "encoder_w_linear_classifier" if model_key == 'linear_encoder' else \
                            "encoder_w_nonlinear_classifier" if model_key == 'nonlinear_encoder' else \
                            "end_to_end_or_feature_classifier"
                
                self.logger.info(f"{epoch_str}{model_name} average accuracy {desc} over {runs} runs: {final_results[model_key]['accuracy'] * 100:.2f}%")
                
                self._log_accuracy(model_name, final_results[model_key]['accuracy'], final_results[model_key]['class_accuracy'], aug_name, epoch=epoch)
            
            return final_results
        
        else:
            # For original data, run once
            desc = "on original data"
            self.logger.info(f"Evaluating classification {desc}...")

            results = {}
        
            # Encoder
            linear_encoder_results = self._evaluate_model_accuracy(
                model=self.linear_encoder_model,
                dataloader=self.testloader,
                augmentation=augmentation,
                model_name="Encoder w/ Linear Classifier",
                aug_name=aug_name
            )
            results['linear_encoder'] = linear_encoder_results

            nonlinear_encoder_results = self._evaluate_model_accuracy(
                model=self.nonlinear_encoder_model,
                dataloader=self.testloader,
                augmentation=augmentation,
                model_name="Encoder w/ Nonlinear Classifier",
                aug_name=aug_name
            )
            results['nonlinear_encoder'] = nonlinear_encoder_results
            
            # End-to-end classifier/feature space classifier (if available)
            if self.end_to_end_classifier:
                end_to_end_results = self._evaluate_model_accuracy(
                    model=self.end_to_end_model,
                    dataloader=self.testloader,
                    augmentation=augmentation,
                    model_name="End-to-End Classifier/Feature Space Classifier",
                    aug_name=aug_name
                )
                results['end_to_end'] = end_to_end_results

            # Log and save results
            for model_key, acc_data in results.items():
                model_name = "encoder_w_linear_classifier" if model_key == 'linear_encoder' else \
                            "encoder_w_nonlinear_classifier" if model_key == 'nonlinear_encoder' else \
                            "end_to_end_or_feature_classifier"
                
                self.logger.info(f"{epoch_str}{model_name} accuracy {desc}: {results[model_key]['accuracy'] * 100:.2f}%")
                
                self._log_accuracy(model_name, results[model_key]['accuracy'], results[model_key]['class_accuracy'], aug_name, epoch=epoch)

            return results
    
    def _evaluate_model_accuracy(self, 
                                 model, 
                                 dataloader, 
                                 augmentation=None, 
                                 model_name="Model", 
                                 aug_name="original"):
        """
        Evaluate classification accuracy for a single model.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for test data
            augmentation: Optional augmentation to apply
            name: Name of the model for logging
            aug_name: Name of the augmentation for logging
            
        Returns:
            Dictionary with accuracy metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        class_correct = np.zeros(self.num_classes)
        class_total = np.zeros(self.num_classes)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                data, targets = batch
            
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                if augmentation is not None:
                    data = augmentation(data)
                
                # Forward pass
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Track per-class statistics
                for i in range(len(targets)):
                    label = targets[i].item()
                    class_correct[label] += (predicted[i] == targets[i]).item()
                    class_total[label] += 1
        
        # Calculate overall accuracy
        accuracy = np.mean([all_predictions[i] == all_targets[i] for i in range(len(all_predictions))])
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                class_accuracy[i] = class_correct[i] / class_total[i]
            else:
                class_accuracy[i] = 0.0
        
        
        return {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy
        }
    

    def _log_accuracy(self, 
                      model_name,
                      accuracy, 
                      class_accuracy, 
                      aug_name='original', 
                      epoch=None):
        epoch_str = f'epoch_{epoch}_' if epoch is not None else ''
        log_file = os.path.join(self.log_dir, f'{epoch_str}classification_accuracy_for_{model_name}_{aug_name}.txt')
        
        if aug_name=='original':
            with open(log_file, 'w') as f:
                f.write(f"Classification Results for {model_name} with augmentation {aug_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"\n{model_name} - Class-wise Accuracy:\n")
                for class_idx, class_acc in class_accuracy.items():
                    f.write(f" Class {class_idx}: {class_acc * 100:.2f}%\n")
                f.write("\n")
                f.write(f"\n{model_name} - Overall Mean Accuracy: {accuracy * 100:.2f}%\n")
                f.write("\n")
        else: 
            with open(log_file, 'w') as f:
                f.write(f"Classification Results for {model_name} with augmentation {aug_name} (averaged over {self.num_classification_runs} runs)\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"\n{model_name} - Class-wise Accuracy:\n")
                for class_idx, class_acc in class_accuracy.items():
                    f.write(f" Class {class_idx}: {class_acc * 100:.2f}%\n")
                f.write("\n")
                f.write(f"\n{model_name} - Overall Mean Accuracy: {accuracy * 100:.2f}%\n")
                f.write("\n")

    
    def _get_class_samples(self):
        self.logger.info("Getting class samples for invariance testing...")
        
        class_samples = defaultdict(list)
        class_counts = {i: 0 for i in range(self.num_classes)}
        
        with torch.no_grad():
            for batch in tqdm(self.testloader):
                data, targets = batch
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Handle stratified sampling
                for i, label in enumerate(targets):
                    label_item = label.item()
                    if class_counts[label_item] < self.num_samples_per_class:
                        # Store individual sample
                        class_samples[label_item].append(data[i:i+1])
                        class_counts[label_item] += 1
                
                if all(count >= self.num_samples_per_class for count in class_counts.values()):
                    break

        for class_idx in class_samples:
            class_samples[class_idx] = class_samples[class_idx][:self.num_samples_per_class]
            class_samples[class_idx] = torch.cat(class_samples[class_idx], dim=0)
        
        self.logger.info(f"Collected {self.num_samples_per_class} samples for each of {self.num_classes} classes")
        
        return class_samples
    
    def measure_invariance(self, 
                           augmentation, 
                           aug_name, 
                           epoch=None):
        """
        Measure invariance properties by computing L2 distances and cosine similarities between 
        original and transformed encodings.
        
        Args:
            augmentation: Augmentation to test
            aug_name: Name of the augmentation for logging
            
        Returns:
            Dictionary of invariance results
        """
        self.logger.info(f"Measuring invariance properties for {aug_name}...")
    
        results = {
            'encoder': {}
        }
    
        # Prepare balanced dataset with samples from each class
        class_samples = self._get_class_samples()
        
        # Measure encoder invariance 
        results['encoder'] = self._measure_model_invariance(
            self.encoder, augmentation, class_samples, 'Encoder', aug_name
        )
        
        # Log results
        self._log_invariance(
            aug_name, 
            results, 
            epoch=None
        )
        
        return results
    
    def _measure_model_invariance(self, 
                                  model, 
                                  augmentation, 
                                  class_samples, 
                                  model_name, 
                                  aug_name):
        """
        Measure invariance.
        
        Args:
            model: Encoder model to evaluate
            augmentation: Augmentation transformation
            class_samples: Dictionary of class samples 
            model_name: Name of model for logging
            aug_name: Name of augmentation for logging
            
        Returns:
            Dictionary of invariance metrics
        """        
        class_mean_distances = {}
        class_mean_cosine_sims = {}

        for class_idx, samples in tqdm(class_samples.items(), 
                                    desc=f"Computing {model_name} invariance for {aug_name}"):
            with torch.no_grad():
                if self.features is not None:
                    features_out = self.features(samples)
                else:
                    features_out = samples
                
                orig_encoded = model(features_out)
                l2_dists = torch.zeros(self.num_samples_per_class, device=self.device)
                cosine_sims = torch.zeros(self.num_samples_per_class, device=self.device)
                
                for _ in range(self.num_aug_samples):
                    aug_batch = augmentation(samples)
                    
                    if self.features is not None:
                        aug_features = self.features(aug_batch)
                    else:
                        aug_features = aug_batch
                    
                    aug_encoded = model(aug_features)
            
                    # Calculate L2 distances
                    l2_dists += torch.norm(aug_encoded - orig_encoded, dim=1)
            
                    # Calculate cosine similarity
                    cosine_sims += F.cosine_similarity(aug_encoded, orig_encoded, dim=1)
                
                class_mean_distances[class_idx] = (torch.sum(l2_dists).cpu()/(self.num_aug_samples * self.num_samples_per_class)).item()
                class_mean_cosine_sims[class_idx] = (torch.sum(cosine_sims).cpu()/(self.num_aug_samples * self.num_samples_per_class)).item()

        mean_distance = np.mean(list(class_mean_distances.values()))
        mean_cosine_sim = np.mean(list(class_mean_cosine_sims.values()))

        results = {
            'class_mean_distances': class_mean_distances,
            'mean_distance': mean_distance,
            'class_mean_cosine_sims': class_mean_cosine_sims,
            'mean_cosine_sim': mean_cosine_sim
        }
        
        return results

    def _log_invariance(self, aug_name, results, epoch=None):
        epoch_str = f'epoch_{epoch}_' if epoch is not None else ''

        log_file = os.path.join(self.log_dir, f'{epoch_str}invariance_{aug_name}.txt')
        
        with open(log_file, 'w') as f:
            f.write(f"Invariance Results for {aug_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Class-wise L2 distances
            f.write("Class-wise L2 Distance Results:\n")
            
            f.write("| Class | Encoder |\n")
            f.write("|-------|-------------------|\n")
            
            for class_idx in sorted(results['encoder']['class_mean_distances'].keys()):
                dist = results['encoder']['class_mean_distances'][class_idx]
                f.write(f"| {class_idx} | {dist:.4f} |\n")
            
            # Class-wise cosine similarities
            f.write("\nClass-wise Cosine Similarity Results:\n")
            
            f.write("| Class | Encoder |\n")
            f.write("|-------|-------------------|\n")
            
            for class_idx in sorted(results['encoder']['class_mean_cosine_sims'].keys()):
                sim = results['encoder']['class_mean_cosine_sims'][class_idx]
                f.write(f"| {class_idx} | {sim:.4f} |\n")

            f.write("\n")
            # Overall results
            f.write("Overall Results:\n")
        
            f.write(f"Mean L2 Distance: {results['encoder']['mean_distance']:.4f}\n")
            f.write(f"Mean Cosine Similarity: {results['encoder']['mean_cosine_sim']:.4f}\n\n")
        

    def run_evaluation(self, tests, epoch):
        """Run the complete evaluation pipeline."""
        self.logger.info("Starting evaluation pipeline...")
    
        if 'classification' and 'invariance' in tests:
            self.results = {
                'classification': {
                    'original': None,
                    'augmented': {}
                },
                'invariance': {}
            }
        elif 'classification' in tests: 
            self.results = {
                'classification': {
                    'original': None,
                    'augmented': {}
                }
            }
        else: 
            self.results = {
                'invariance': {}
            }
        
        if 'classification' in tests:
            # Evaluate on original data (no augmentation)
            self.logger.info("Evaluating on original data...")

            self.train_classifiers() 
            self.results['classification']['original'] = self.evaluate_classification(epoch=epoch)
            
            # For each augmentation, run both classification and invariance tests
            if self.augmentations is not None:
                for aug_name, augmentation in self.augmentations.items():
                    self.logger.info(f"Evaluating augmentation: {aug_name}")
                    
                    # Evaluate classification performance with this augmentation
                    self.results['classification']['augmented'][aug_name] = self.evaluate_classification(
                        augmentation=augmentation,
                        aug_name=aug_name, 
                        epoch=epoch
                    )

        # Measure invariance properties
        if 'invariance' in tests:
            for aug_name, augmentation in self.augmentations.items():
                self.results['invariance'][aug_name] = self.measure_invariance(
                    augmentation=augmentation,
                    aug_name=aug_name, 
                    epoch=epoch
                    )

        # Save summary of results
        summary_report = os.path.join(self.log_dir, f"summary_report_epoch_{epoch}.txt")

        with open(summary_report, 'w') as f:
            if 'classification' in tests:
                f.write(f"Invariance Evaluation\n")
                f.write("=" * 60 + "\n\n")
                
                # Original test set performance
                original = self.results['classification']['original']
                f.write("Classification on Original Test Data:\n")
                f.write("-" * 40 + "\n")
                
                f.write(f"Encoder w/ Linear Classifier: {original['linear_encoder']['accuracy'] * 100:.2f}%\n")

                f.write(f"Encoder w/ Nonlinear Classifier: {original['nonlinear_encoder']['accuracy'] * 100:.2f}%\n")
                
                if 'end_to_end' in original:
                    f.write(f"End-to-End Model/Feature Classifier: {original['end_to_end']['accuracy'] * 100:.2f}%\n")
                f.write("\n")
                
                if self.augmentations is not None:
                    # Augmented test set performance
                    f.write(f"Classification on Augmented Test Data (averaged over {self.num_classification_runs} runs):\n")
                    f.write("-" * 40 + "\n")
                    
                    for aug_name, aug_results in self.results['classification']['augmented'].items():
                        f.write(f"\n{aug_name}:\n")
                        
                        f.write(f"  Encoder w/ Linear Classifier: {aug_results['linear_encoder']['accuracy'] * 100:.2f}%\n")
                        f.write(f"  Encoder w/ Nonlinear Classifier: {aug_results['nonlinear_encoder']['accuracy'] * 100:.2f}%\n")

                        
                        if 'end_to_end' in aug_results:
                            f.write(f"  End-to-End Model/Feature Classifier: {aug_results['end_to_end']['accuracy'] * 100:.2f}%\n")
                
                f.write("\n")

            # Invariance measurements
            if 'invariance' in tests:
                f.write("Invariance Measurements:\n")
                f.write("-" * 40 + "\n")
                
                for aug_name, aug_results in self.results['invariance'].items():
                    f.write(f"\n{aug_name}:\n")
                    
                    f.write(f"  Encoder Mean L2 Distance: {aug_results['encoder']['mean_distance']:.4f}\n")
                
                    f.write(f"  Encoder Mean Cosine Similarity: {aug_results['encoder']['mean_cosine_sim']:.4f}\n")
                    
                f.write("\n")

        self.logger.info(f"Summary report saved to {self.log_dir}")
        self.logger.info("Evaluation complete.")

        return self.results

    def _create_save_directories(self):
        self.evaluate_dir = os.path.join(self.save_dir, 'evaluate')
        os.makedirs(self.evaluate_dir, exist_ok=True)  
        self.log_dir = os.path.join(self.evaluate_dir, 'invariance_tests')
        os.makedirs(self.log_dir, exist_ok=True)  

        self.log_file = os.path.join(self.log_dir, 'training_and_evaluation.log')
        

    def _configure_logging(self):
        """Configure logging"""
        logger = logging.getLogger()
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        self.logger = logger
        
        logging.info(f"Logging configured to file: {self.log_file}")
    
    def _get_max_samples(self, dataloader):
        dataset = dataloader.dataset
        class_counts = {}
        
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                labels = targets.cpu().numpy()
                unique_labels, counts = np.unique(labels, return_counts=True)
                class_counts = {int(label): count for label, count in zip(unique_labels, counts)}
            elif isinstance(targets, (list, np.ndarray)):
                unique_labels, counts = np.unique(targets, return_counts=True)
                class_counts = {int(label) if hasattr(label, 'item') else label: count 
                            for label, count in zip(unique_labels, counts)}
        else:
            for i in range(len(dataset)):
                item = dataset[i]
                if isinstance(item, tuple) and len(item) >= 2:
                    label = item[1]
                elif hasattr(item, 'label'):
                    label = item.label
                else:
                    raise ValueError("Cannot determine label from dataset item.")
                
                if hasattr(label, 'item'):
                    label = label.item()
                    
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
        
        # Calculate max_samples (minimum count across classes)
        max_samples = min(class_counts.values()) if class_counts else 0
        
        return max_samples