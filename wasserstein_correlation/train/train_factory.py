import torch
from torch import nn
import inspect
import copy 
import ast
from torch.utils.data import DataLoader
from wasserstein_correlation.models.model_factory import ModelFactory
from wasserstein_correlation.config.model_config import ModelConfig
from wasserstein_correlation.losses.loss_factory import LossFactory
from wasserstein_correlation.config.train_config import TrainConfig
from wasserstein_correlation.config.dataloader_config import DataloaderConfig
from wasserstein_correlation.train.training_components import TrainingComponents
from wasserstein_correlation.utils.create_dataset import create_dataset

class TrainFactory:
    @staticmethod
    def create_training_components(config: TrainConfig):
        trainset, testset = create_dataset(config.run_config.dataset, config.run_config.root_dir)
        invariance_trainset, invariance_testset = create_dataset(config.run_config.invariance_dataset, config.run_config.root_dir)

        config.trainloader_config.dataset = trainset
        config.testloader_config.dataset = testset

        TrainFactory.set_seed(config.run_config.seed)
        batch_size = config.run_config.batch_size
        config.trainloader_config.params['batch_size'] = batch_size

        if not config.testloader_config.params.get('batch_size', None):
            config.testloader_config.params['batch_size'] = batch_size

        if 'generator' in config.trainloader_config.params:
            traingenerator = config.trainloader_config.params['generator']
            if config.run_config.seed is not None:
                traingenerator.manual_seed(config.run_config.seed)
        else:
            traingenerator = torch.Generator()
            if config.run_config.seed is not None:
                traingenerator.manual_seed(config.run_config.seed)
            config.trainloader_config.params['generator'] = traingenerator

        trainloader = TrainFactory.create_dataloader(trainset, config.trainloader_config.params)
        testloader = TrainFactory.create_dataloader(testset, config.testloader_config.params)

        if config.features_config is not None and isinstance(config.features_config, ModelConfig): 
            features = ModelFactory.create_model_config(config.features_config)
        elif config.features_config is not None and isinstance(config.features_config, nn.Module): 
            features = config.features_config
        else:  
            features = None

        if isinstance(config.encoder_config, ModelConfig): 
            encoder = ModelFactory.create_model_config(config.encoder_config)
        elif isinstance(config.encoder_config, nn.Module):
            encoder = config.encoder_config

        if config.decoder_config is not None and isinstance(config.decoder_config, ModelConfig): 
            decoder = ModelFactory.create_model_config(config.decoder_config)
        elif config.decoder_config is not None and isinstance(config.decoder_config, nn.Module): 
            decoder = config.decoder_config
        else: 
            decoder = None
        
        augmentations = config.augmentations
        
        loss_fn = LossFactory.create_loss(config.loss_config)
        
        optimizer = TrainFactory.create_optimizer(encoder, decoder, config.optimizer_config)
        
        if config.scheduler_config is not None:
            # Dynamically configure scheduler parameters 
            dynamic_values = {}

            if any(isinstance(v, str) and 'steps_per_epoch' in v for v in config.scheduler_config.params.values()):
                steps_per_epoch = TrainFactory.calculate_steps_per_epoch(
                    config.trainloader_config.dataset, 
                    batch_size, 
                    config.trainloader_config.params.get('drop_last', True)
                )
                dynamic_values['steps_per_epoch'] = steps_per_epoch

            if any(isinstance(v, str) and 'epochs' in v for v in config.scheduler_config.params.values()):
                dynamic_values['epochs'] = config.run_config.epochs
            
            if any(isinstance(v, str) and 'total_steps' in v for v in config.scheduler_config.params.values()):
                total_steps = len(trainloader) * config.run_config.epochs
                dynamic_values['total_steps'] = total_steps

            for key, value in dynamic_values.items():
                config.scheduler_config.params = TrainFactory.dynamic_params_replacement(
                    config.scheduler_config.params, 
                    key, 
                    value
                )

            scheduler = TrainFactory.create_scheduler(optimizer, config.scheduler_config)
        else: 
            scheduler = None
        
        invariance_config = copy.deepcopy(config.invariance_config)
        if invariance_config is not None:
            if isinstance(invariance_config.invariance_trainloader, DataloaderConfig):
                config.invariance_config.invariance_trainloader.dataset = invariance_trainset
                invariance_config.invariance_trainloader = TrainFactory.create_dataloader(invariance_trainset, invariance_config.invariance_trainloader.params)
            if isinstance(invariance_config.invariance_testloader, DataloaderConfig):
                config.invariance_config.invariance_testloader.dataset = invariance_testset
                invariance_config.invariance_testloader = TrainFactory.create_dataloader(invariance_testset, invariance_config.invariance_testloader.params)
    
        return TrainingComponents(
            run_config=config.run_config,
            trainloader=trainloader,
            testloader=testloader,
            features=features,
            encoder=encoder, 
            decoder=decoder,
            augmentations=augmentations,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler, 
            invariance_config=invariance_config
        )

    @staticmethod
    def set_seed(seed):
        if seed is not None:
            print(f"Setting random seed: {seed}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False  
        else:
            print("No seed provided. Proceeding without.")
            

    @staticmethod
    def create_dataloader(dataset, dataloader_config_params):
        valid_params = TrainFactory.filter_parameters(DataLoader, dataloader_config_params)

        return DataLoader(dataset, **valid_params)
    
    @staticmethod
    def create_optimizer(encoder, decoder=None, optimizer_config=None):
        modules = {
            'encoder': encoder,
            'decoder': decoder,
        }
        
        params = []
        for _, module in modules.items():
            if module is not None:
                params.extend(module.parameters())
        
        # Get optimizer class and valid parameters
        optimizer_class = optimizer_config.optimizer_class
        valid_params = TrainFactory.filter_parameters(optimizer_class, optimizer_config.params)
        
        return optimizer_class(params, **valid_params)

    
    @staticmethod
    def create_scheduler(optimizer, scheduler_config):
        scheduler_class = scheduler_config.scheduler_class
        valid_params = TrainFactory.filter_parameters(scheduler_class, scheduler_config.params)

        return scheduler_class(optimizer, **valid_params)

    @staticmethod
    def calculate_steps_per_epoch(dataset, batch_size, drop_last=True):
        if drop_last:
            steps_per_epoch = len(dataset) // batch_size
        else:
            steps_per_epoch = (len(dataset) + batch_size - 1) // batch_size  

        return steps_per_epoch

    @staticmethod
    def dynamic_params_replacement(params, value_string, value_literal):
        """
        Replaces dynamic parameter references in config['params']
        """
        
        for key, value in params.items():
            if isinstance(value, str) and value_string in value:
                try:
                    parsed_value = ast.parse(value, mode='eval')
                    evaluated_value = eval(compile(parsed_value, '', mode='eval'), {value_string: value_literal})
                    params[key] = evaluated_value

                except Exception as e:
                    print(f"Warning: Couldn't evaluate expression for key '{key}' with value '{value}'. Skipping. Error: {e}")
    
        return params

    @staticmethod
    def filter_parameters(class_or_function, params):
        """
        Filter parameter dictionary 
        """
        if inspect.isclass(class_or_function):
            signature = inspect.signature(class_or_function.__init__)
        else:
            signature = inspect.signature(class_or_function)
        
        valid_params = {k: v for k, v in params.items() if k in signature.parameters}

        return valid_params
