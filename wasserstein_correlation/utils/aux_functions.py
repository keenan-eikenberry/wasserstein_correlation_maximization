import os
import torch
import numpy as np
import torchvision.transforms as transforms
import inspect 

def set_root_directory(root_dir): 
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"The directory '{root_dir}' does not exist.")
    
    current_dir = os.getcwd()

    if current_dir != root_dir:
        os.chdir(root_dir)
        print(f"Changed working directory to: {root_dir}")
    else:
        print(f"Current working directory is already: {root_dir}")


def extract_normalization(dataset):
    if dataset.transform:
        transform = dataset.transform

        if isinstance(transform, transforms.Normalize):
            return np.array(transform.mean), np.array(transform.std)

        if isinstance(transform, transforms.Compose):
            for t in transform.transforms:  
                if isinstance(t, transforms.Normalize):
                    return np.array(t.mean), np.array(t.std)
                
    sample_data = dataset[0]
    if isinstance(sample_data, (list, tuple)): 
        sample_data = sample_data[0]

    channels = sample_data.shape[0]
   
    # Default normalization 
    return np.array([0.0] * channels), np.array([1.0] * channels)


def denormalize(image, mean, std):
    if isinstance(mean, np.ndarray):
        mean = torch.tensor(mean, dtype=image.dtype, device=image.device)
    if isinstance(std, np.ndarray):
        std = torch.tensor(std, dtype=image.dtype, device=image.device)
    
    if image.dim() == 3:  # [C, H, W]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif image.dim() == 4:  # [B, C, H, W]
        mean = mean.view(1, -1, 1, 1)   
        std = std.view(1, -1, 1, 1)
    
    denormalized = image * std + mean
    
    denormalized = torch.clamp(denormalized, 0.0, 1.0)
    
    return denormalized


def create_save_directories(root_dir, save_dir_prefix):
    """
    Creates a new experiment directory under `root_dir' with the provided prefix.
    Folders are numbered incrementally based on existing ones.
    
    Args:
        root_dir (str): The root directory where the `results` folder exists.
        prefix (str): A string prefix for the experiment name (e.g., 'STL10_invariance_test').
    """
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract all numbers from existing directories with the given prefix
    experiment_dirs = [d for d in os.listdir(results_dir) if d.startswith(save_dir_prefix)]
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
        experiment_name = f"{save_dir_prefix}_{experiment_number:03d}"
        save_dir = os.path.join(results_dir, experiment_name)
        
        if not os.path.exists(save_dir):
            break
        experiment_number += 1
    
    # Create subdirectories
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    evaluate_dir = os.path.join(save_dir, 'evaluate')
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(evaluate_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)  
    
    print(f"Created experiment directory: {save_dir}")

    return save_dir, evaluate_dir, experiment_name


def filter_parameters(class_or_function, params):
        """
        Filter dictionary
        """
        if inspect.isclass(class_or_function):
            signature = inspect.signature(class_or_function.__init__)
        else:
            signature = inspect.signature(class_or_function)
        
        valid_params = {k: v for k, v in params.items() if k in signature.parameters}

        return valid_params
