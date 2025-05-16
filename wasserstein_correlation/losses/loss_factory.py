import torch
from typing import Optional, Callable
from wasserstein_correlation.config.loss_config import LossConfig
from wasserstein_correlation.config.run_config import RunConfig
from wasserstein_correlation.losses.wasserstein import sliced_wasserstein_correlation, sliced_wasserstein_correlation_vectorized


class LossFactory:
    @staticmethod 
    def create_loss(loss_config: LossConfig) -> Callable:
        
        def loss_function(
            data: torch.Tensor,
            encoded: torch.Tensor,
            augmented_encoded: Optional[torch.Tensor],
            decoded: Optional[torch.Tensor], 
            run_config: RunConfig,      
            ) -> torch.Tensor:

            total_loss = torch.zeros(1, device=data.device)
            loss_terms = {}

            num_projections = run_config.num_wasserstein_projections
            p = run_config.wasserstein_p
            q = run_config.wasserstein_q
            standardize = run_config.standardize_correlation
            device = run_config.set_gpu 

            if augmented_encoded is not None:
                num_augs = run_config.num_augs
                num_aug_samples = run_config.num_aug_samples
                log_product_loss = run_config.log_product_loss

            # Flatten
            flat_data = data.flatten(1)
            flat_encoded = encoded.flatten(1)

            if augmented_encoded is not None:
                if not log_product_loss: 
                    flat_augmented_encoded = augmented_encoded.flatten(1)
                else:
                    flat_augmented_encoded = augmented_encoded.flatten(2, -1)      

            if decoded is not None: 
                flat_decoded = decoded.flatten(1)

            # Reconstruction loss 
            if loss_config.reconstruct > 0:
                reconstruction_error = torch.mean(torch.square(flat_data - flat_decoded))

                total_loss += loss_config.reconstruct * reconstruction_error
                loss_terms['L2_reconstruction_error'] = reconstruction_error

            # Wasserstein correlation; optionally take log product of correlation scores in case of multiple augmentations 
            if loss_config.wasserstein_correlation > 0:
                if augmented_encoded is None: 
                    X = flat_data 
                    Y = flat_encoded
                    wasserstein_correlation = sliced_wasserstein_correlation(flat_data, flat_encoded, num_projections=num_projections, p=p, q=q, standardize=standardize, device=device)

                    total_loss -= loss_config.wasserstein_correlation * wasserstein_correlation 
                    loss_terms['wasserstein_correlation'] = wasserstein_correlation
                else:
                    if not log_product_loss:
                        X = torch.cat([flat_data] * (num_aug_samples*num_augs + 1), dim=0) 
                        Y = torch.cat([flat_encoded, flat_augmented_encoded], dim=0)
                        wasserstein_correlation = sliced_wasserstein_correlation(X, Y, num_projections=num_projections, p=p, q=q, standardize=standardize, device=device)
            
                        total_loss -= loss_config.wasserstein_correlation * wasserstein_correlation 
                        loss_terms['wasserstein_correlation'] = wasserstein_correlation
                    else:
                        X = torch.cat([flat_data] * (num_aug_samples + 1), dim=0) 
                        X = X.unsqueeze(1).expand(-1, num_augs, -1)
                        encoded_expand = flat_encoded.unsqueeze(1).expand(-1, num_augs, -1)
                        Y = torch.cat([encoded_expand, flat_augmented_encoded], dim=0)
                        wasserstein_correlation_vec = sliced_wasserstein_correlation_vectorized(X, Y, num_projections=num_projections, p=p, q=q, standardize=standardize, device=device)

                        log_product_wasserstein_correlation = torch.sum(torch.log(wasserstein_correlation_vec))

                        total_loss -= loss_config.wasserstein_correlation * log_product_wasserstein_correlation 
                        loss_terms['log_product_wasserstein_correlation'] = log_product_wasserstein_correlation

            return total_loss, loss_terms

        return loss_function
