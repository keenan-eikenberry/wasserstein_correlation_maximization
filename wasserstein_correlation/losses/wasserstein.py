import torch

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))

    return projections


def sliced_wasserstein(X, Y, num_projections=1000, p=2.0, device='cuda'):
    dim = X.size(1)
    theta = rand_projections(dim, num_projections).to(device)
    X_proj = torch.matmul(X, theta.transpose(0, 1))
    Y_proj = torch.matmul(Y, theta.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_proj, dim=0)[0]
                - torch.sort(Y_proj, dim=0)[0]
        )
    )
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    wasserstein_distance = torch.pow(torch.mean(wasserstein_distance), 1.0 / p)

    return wasserstein_distance


def sliced_wasserstein_vectorized(X, Y, num_projections=1000, p=2.0, device='cuda'):
    """
    Compute the Sliced Wasserstein distance between two distributions X and Y in a vectorized manner.
    X and Y are tensors of shape (batch_size, num_augs, D).
    Returns a tensor of shape (num_augs,)
    """
    batch_size, num_patches, D = X.shape

    # Generate unique random projections for each augmentation
    # Shape: (num_augs, L, D)
    theta = torch.randn(num_patches, num_projections, D, device=device)
    theta = theta / torch.norm(theta, dim=2, keepdim=True) 

    # X and Y: (batch_size, num_augs, D) -> (num_augs, batch_size, D)
    X = X.permute(1, 0, 2)
    Y = Y.permute(1, 0, 2)

    # Project the distributions
    # X_proj and Y_proj: (num_augs, batch_size, num_projections)
    X_proj = torch.matmul(X, theta.transpose(1, 2))
    Y_proj = torch.matmul(Y, theta.transpose(1, 2))

    # Sort over batch_size dimension
    X_proj_sorted, _ = torch.sort(X_proj, dim=1)
    Y_proj_sorted, _ = torch.sort(Y_proj, dim=1)

    # Compute the Wasserstein distance
    wasserstein_distance = torch.abs(X_proj_sorted - Y_proj_sorted)
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    wasserstein_distance = torch.mean(wasserstein_distance, dim=1)
    wasserstein_distance = torch.mean(wasserstein_distance, dim=1)  
    wasserstein_distance = torch.pow(wasserstein_distance, 1.0 / p)  

    return wasserstein_distance  


def sliced_wasserstein_correlation(
    X, Y, num_projections=1000, p=2.0, q=2.0, standardize=False, eps=1e-10, device='cuda'):
    if standardize:
        X_mean = torch.mean(X, dim=0, keepdim=True)
        X_std = torch.std(X, dim=0, unbiased=False, keepdim=True)
        X_std = X_std + eps 
        X = (X - X_mean) / X_std

        Y_mean = torch.mean(Y, dim=0, keepdim=True)
        Y_std = torch.std(Y, dim=0, unbiased=False, keepdim=True)
        Y_std = Y_std + eps
        Y = (Y - Y_mean) / Y_std

    batch_size = X.size(0)
    
    # Joint distribution samples (X_i, Y_i)
    joint = torch.cat((X, Y), dim=1)
    
    # Product of marginals samples (X_i, Y_j) with shuffled Y
    X_shuffled = X[torch.randperm(batch_size)]
    Y_shuffled = Y[torch.randperm(batch_size)]
    prod = torch.cat((X_shuffled, Y_shuffled), dim=1)
    
    # SWD between joint and product of marginals
    dependence = sliced_wasserstein(joint, prod, num_projections, p, device)
    
    # SWD for X
    X_joint = torch.cat((X, X), dim=1)
    X_shuffled = X[torch.randperm(batch_size)]
    X_prod = torch.cat((X, X_shuffled), dim=1)
    X_dependence = sliced_wasserstein(X_joint, X_prod, num_projections, p, device)
    
    # SWD for Y
    Y_joint = torch.cat((Y, Y), dim=1)
    Y_shuffled = Y[torch.randperm(batch_size)]
    Y_prod = torch.cat((Y, Y_shuffled), dim=1)
    Y_dependence = sliced_wasserstein(Y_joint, Y_prod, num_projections, p, device)
    
    normalization = (X_dependence * Y_dependence + eps)**(1/q)
    SW_correlation = dependence / normalization

    return SW_correlation


def sliced_wasserstein_correlation_vectorized(
    X, Y, num_projections=1000, p=2.0, q=2.0, standardize=False, eps=1e-10, device='cuda'):
    """
    Compute the Sliced Wasserstein Correlation between two distributions X and Y in a vectorized manner.
    X and Y are tensors of shape (batch_size, num_augs, D_1) and (batch_size, num_augs, D_2), respectively.
    Returns a tensor of shape (num_augs,)
    """
    batch_size, num_augs, D = X.shape

    if standardize:
        # Standardize X and Y over the batch dimension
        X_mean = torch.mean(X, dim=0, keepdim=True) 
        X_std = torch.std(X, dim=0, unbiased=False, keepdim=True) + eps
        X = (X - X_mean) / X_std

        Y_mean = torch.mean(Y, dim=0, keepdim=True)
        Y_std = torch.std(Y, dim=0, unbiased=False, keepdim=True) + eps
        Y = (Y - Y_mean) / Y_std

    joint = torch.cat((X, Y), dim=2)  

    Y_shuffled = Y.permute(1, 0, 2)
    Y_shuffled = Y_shuffled[torch.arange(num_augs).unsqueeze(1), torch.stack([torch.randperm(batch_size) for _ in range(num_augs)])]
    Y_shuffled = Y_shuffled.permute(1, 0, 2)

    prod = torch.cat((X, Y_shuffled), dim=2)  # Shape: (batch_size, num_augs, 2D)

    # Compute sliced Wasserstein distance between joint and product of marginals
    dependence = sliced_wasserstein_vectorized(joint, prod, num_projections, p, device)

    X_joint = torch.cat((X, X), dim=2)
    X_shuffled = X.permute(1, 0, 2)
    X_shuffled = X_shuffled[torch.arange(num_augs).unsqueeze(1), torch.stack([torch.randperm(batch_size) for _ in range(num_augs)])]
    X_shuffled = X_shuffled.permute(1, 0, 2)
    X_prod = torch.cat((X, X_shuffled), dim=2)
    X_dependence = sliced_wasserstein_vectorized(X_joint, X_prod, num_projections, p, device)

    Y_joint = torch.cat((Y, Y), dim=2)
    Y_shuffled = Y.permute(1, 0, 2)
    Y_shuffled = Y_shuffled[torch.arange(num_augs).unsqueeze(1), torch.stack([torch.randperm(batch_size) for _ in range(num_augs)])]
    Y_shuffled = Y_shuffled.permute(1, 0, 2)
    Y_prod = torch.cat((Y, Y_shuffled), dim=2)
    Y_dependence = sliced_wasserstein_vectorized(Y_joint, Y_prod, num_projections, p, device)

    # Compute the normalization term and the sliced Wasserstein correlation
    normalization = (X_dependence * Y_dependence + eps)**(1/q)
    SW_correlation = dependence / normalization

    return SW_correlation  # Shape: (num_augs,)
