from wasserstein_correlation.utils.common_imports import *

def main():
    root_dir = 'default_root_dir'
    
    # Training configuration
    configs = [
       TrainConfig(
        run_config=RunConfig(
            root_dir=root_dir,
            dataset='STL10_Features',
            invariance_dataset='STL10_Features_Classify', 
            save_dir_prefix='STL10_rotation_test',
            batch_size=256,
            epochs=100, 
            checkpoint_epoch=10, 
            evaluate_structure_epoch=100,
            # Structure tests ['topological', 'spectral', 'heat']
            # Use 'tsne_all' for both input and encoded distribution; 'tsne' for encoded distribution only 
            structure_tests=['topological', 'spectral', 'heat', 'tsne_all'],
            evaluate_invariance_epoch=100, 
            # Invariance tests ['classification', 'invariance']
            invariance_tests=['classification'],
            num_wasserstein_projections=1000,
            wasserstein_p=2.0,
            wasserstein_q=2.0,
            standardize_correlation=False,
            num_augs=1,
            num_aug_samples=3,
            log_product_loss=False, 
            seed=12, 
            load_dir=None, 
            checkpoint_filename=None, 
            set_gpu='cuda',
            # Set scheduler_batch_update to None for epoch updates 
            # Set to (k - 1) to update every k batches 
            scheduler_batch_update=None, 
            use_autocast=False,
            grad_set_to_none=False,
            clear_cache_batch=None,
            clear_cache_epoch=None,
            print_memory_and_optimizer_stats=False, 
            training_run_info=None
        ), 

        trainloader_config=DataloaderConfig(
            dataset='train', 
            params={
                'shuffle': True,
                'drop_last': True, 
                'num_workers': 4
            }
        ), 

        testloader_config=DataloaderConfig(
            dataset='test', 
            params={
                'batch_size': 1000,
                'shuffle': True,
                'drop_last': False,
                'num_workers': 0
            }
        ),

        features_config=DINO_ViTs8_Features(),

        encoder_config=ModelConfig(
            model_class=Model.MLP,
            params={
                'input_shape': (384,),
                'output_shape': (92,),  
                'dense_features': [384, 384, 192, 92],  
                'normalization_layer': None,
                'activation': 'relu'
            }
        ),      

        decoder_config=None,
       
        augmentations={
            't_1': K.RandomRotation(degrees=180, p=1.0, same_on_batch=False)
        }, 

        loss_config=LossConfig(
            reconstruct=0.0,
            wasserstein_correlation=1.0
        ),

        optimizer_config = OptimizerConfig(
            optimizer_class=optim.AdamW,
            params={
                'lr': 0.001, 
                'weight_decay': 0.0001
            }
        ),
 
        scheduler_config = SchedulerConfig(
            scheduler_class=optim.lr_scheduler.CosineAnnealingLR,
            params={
                'T_max': 'epochs', 
                'eta_min': 0.0004
            }
        ),

        invariance_config=InvarianceConfig(
            num_classes=10,

            augmentations = {
                't_1': K.RandomRotation(degrees=180, p=1.0, same_on_batch=False)
            }, 

            invariance_trainloader=DataloaderConfig(
            dataset='train', 
            params={
                'batch_size': 256,
                'shuffle': True,
                'drop_last': True, 
                'num_workers': 4
            }
            ), 

            invariance_testloader=DataloaderConfig(
                dataset='test', 
                params={
                    'batch_size': 1000,
                    'shuffle': True,
                    'drop_last': False,
                    'num_workers': 0
                }
            ),

            classifier_epochs=50, 

            linear_classifier_config=ModelConfig(
                model_class=Model.MLP,
                params={
                    'input_shape': (92,),
                    'output_shape': (10,),  
                    'dense_features': [92, 10],  
                    'normalization_layer': None,
                    'activation': 'relu'
                }
            ),

            nonlinear_classifier_config=ModelConfig(
                model_class=Model.MLP,
                params={
                    'input_shape': (92,),
                    'output_shape': (10,),  
                    'dense_features': [92, 92, 10],  
                    'normalization_layer': None,
                    'activation': 'relu'
                }
            ),

            classifier_optimizer_config = OptimizerConfig(
                optimizer_class=optim.AdamW,
                params={
                    'lr': 0.001,
                    'weight_decay': 0.0  
                }
            ),

            classifier_scheduler_config = SchedulerConfig(
                scheduler_class=lr_scheduler.CosineAnnealingLR,
                params={
                    'T_max': 50, 
                    'eta_min': 0.0004
                }
            ),

            # End-to-end, fully supervised classifier
            end_to_end_classifier_epochs = 50,

            end_to_end_classifier_config=ModelConfig(
                model_class=Model.MLP,
                params={
                    'input_shape': (384,),
                    'output_shape': (10,),  
                    'dense_features': [384, 384, 192, 96, 10],  
                    'use_batch_norm': False,
                    'activation': 'relu'
                }
            ),

            end_to_end_optimizer_config = OptimizerConfig(
                optimizer_class=optim.AdamW,
                params={
                    'lr': 0.001, 
                    'weight_decay': 0.0
                }
            ),

            end_to_end_scheduler_config = SchedulerConfig(
                scheduler_class=lr_scheduler.CosineAnnealingLR,
                params={
                    'T_max': 50, 
                    'eta_min': 0.0004
                }
            )
        )
    )
    ]

    # Start training
    for i, _ in enumerate(configs):
        trainer = Trainer(configs[i])
        trainer.train()  
        del trainer
    
if __name__ == "__main__":
    main()