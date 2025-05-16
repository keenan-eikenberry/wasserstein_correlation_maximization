import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, STL10
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms.functional as TF


class RotatedDataset(Dataset):
    def __init__(self, dataset, rotation_angle=90, label_offset=10):
        self.dataset = dataset
        self.rotation_angle = rotation_angle
        self.label_offset = label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Rotate the image by the specified angle
        img = TF.rotate(img, angle=self.rotation_angle)
        # Adjust the label
        label = label + self.label_offset
        return img, label


def create_dataset(dataset_name: str, root_dir: str='default_root_dir'): 
    """
    Create train and test datasets.
    """
    
    data_dir = os.path.join(root_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name == 'MNIST':
        # dimensions: (N, 1, 28, 28), flattened: 784
        # train_size: 60000, test_size: 10000
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        trainset = MNIST(data_dir, train=True, download=True, transform=transform)
        testset = MNIST(data_dir, train=False, download=True, transform=transform)
        
        return trainset, testset

    elif dataset_name == 'MNIST_Rotated_Full':     
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
            )
        
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
            )
        
        trainset = MNIST(data_dir, train=True, download=True, transform=train_transform)
        testset = MNIST(data_dir, train=False, download=True, transform=test_transform)

        trainset_rotated = RotatedDataset(trainset)
        testset_rotated = RotatedDataset(testset)

        trainset_full = ConcatDataset([trainset, trainset_rotated])
        testset_full = ConcatDataset([testset, testset_rotated])

        classes_rotated = [i+10 for i in range(10)]
        classes_full = [i for i in range(10)] + [i+10 for i in range(10)]

        # Set classes
        trainset_rotated.classes = classes_rotated
        testset_rotated.classes = classes_rotated
        trainset_full.classes = classes_full
        testset_full.classes = classes_full

        return trainset, testset, trainset_rotated, testset_rotated, trainset_full, testset_full
    
    elif dataset_name == 'MNIST_Rotated':     
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
            )
        
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
            )
        
        trainset = MNIST(data_dir, train=True, download=True, transform=train_transform)
        testset = MNIST(data_dir, train=False, download=True, transform=test_transform)

        trainset_rotated = RotatedDataset(trainset)
        testset_rotated = RotatedDataset(testset)

        trainset_full = ConcatDataset([trainset, trainset_rotated])
        testset_full = ConcatDataset([testset, testset_rotated])

        classes_rotated = [i+10 for i in range(10)]
        classes_full = [i for i in range(10)] + [i+10 for i in range(10)]

        # Set classes
        trainset_rotated.classes = classes_rotated
        testset_rotated.classes = classes_rotated
        trainset_full.classes = classes_full
        testset_full.classes = classes_full

        return trainset, testset_full
    
    elif dataset_name == 'CIFAR10':
        # dimensions: (N, 3, 32, 32), flattened: 3072
        # train_size: 50000, test_size: 10000
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.247, 0.243, 0.261]
            )   
        ])
        
        trainset = CIFAR10(data_dir, train=True, download=True, transform=transform)
        testset = CIFAR10(data_dir, train=False, download=True, transform=transform)
        
        return trainset, testset
    
    elif dataset_name == 'CIFAR10_Features':
        # dimensions: (N, 3, 32, 32), flattened: 3072
        # train_size: 50000, test_size: 10000
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        trainset = CIFAR10(data_dir, train=True, download=True, transform=transform)
        testset = CIFAR10(data_dir, train=False, download=True, transform=transform)
        
        return trainset, testset
    
    elif dataset_name == 'STL10':
        # dimensions: (N, 3, 96, 96), flattened: 27648
        # unlabeled: 100000, train: 5000, test: 8000
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4467, 0.4398, 0.4066],
                std=[0.2603, 0.2566, 0.2713]
            )
        ])
        
        trainset = STL10(data_dir, split='unlabeled', download=True, transform=transform)
        testset = STL10(data_dir, split='test', download=True, transform=transform)
            
        return trainset, testset
    
    elif dataset_name == 'STL10_Features':
        # dimensions: (N, 3, 96, 96), flattened: 27648
        # unlabeled: 100000, train: 5000, test: 8000
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        trainset = STL10(data_dir, split='unlabeled', download=True, transform=transform)
        testset = STL10(data_dir, split='test', download=True, transform=transform)
            
        return trainset, testset
    
    elif dataset_name == 'STL10_Features_Classify':
        # dimensions: (N, 3, 96, 96), flattened: 27648
        # unlabeled: 100000, train: 5000, test: 8000
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        trainset = STL10(data_dir, split='train', download=True, transform=transform)
        testset = STL10(data_dir, split='test', download=True, transform=transform)
            
        return trainset, testset
    

class GaussianMixtureDataset(Dataset):
    def __init__(self, n_samples, n_components, n_features, means=None, covariances=None, weights=None, transform=None, device='cpu'):
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_features = n_features
        self.transform = transform
        self.device = device

        # Set default means if not provided
        if means is None:
            self.means = torch.empty(n_components, n_features, device=device).uniform_(-10, 10)
        else:
            self.means = means.to(device)

        # Set default covariances if not provided
        if covariances is None:
            self.covariances = torch.stack([torch.eye(n_features, device=device) for _ in range(n_components)])
        else:
            if covariances.dim() == 2:  # Diagonal elements provided
                self.covariances = torch.stack([torch.diag(covariances[i]) for i in range(n_components)])
            elif covariances.dim() == 3:
                self.covariances = covariances.to(device)
            else:
                raise ValueError("Covariances must be either 2D (diagonal elements) or 3D (full matrices).")

        # Set default weights if not provided
        if weights is None:
            self.weights = torch.ones(n_components, device=device) / n_components
        else:
            self.weights = weights.to(device)
            self.weights = self.weights / self.weights.sum()  # Normalize weights

        # Store component parameters in dictionary
        self.component_params = {}
        for idx in range(self.n_components):
            self.component_params[idx] = {
                'mean': self.means[idx],
                'covariance': self.covariances[idx],
                'weight': self.weights[idx]
            }

        # Generate samples
        self.raw_data, self.labels = self._generate_samples()

        # Normalize the data and store mean and std
        self.data_mean = self.raw_data.mean(dim=0)
        self.data_std = self.raw_data.std(dim=0)
        self.data_std[self.data_std == 0] = 1.0  
        self.data = (self.raw_data - self.data_mean) / self.data_std

    def _generate_samples(self):
        data = []
        labels = []

        # Sample component indices for each data point
        component_indices = torch.multinomial(self.weights, self.n_samples, replacement=True)

        # Generate samples for each component
        for idx in range(self.n_components):
            # Get indices where the component matches idx
            idx_mask = (component_indices == idx)
            n_samples = idx_mask.sum().item()

            if n_samples > 0:
                mean = self.means[idx]
                covariance_matrix = self.covariances[idx]

                # Create distribution
                distribution = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance_matrix)

                # Sample n_samples points
                samples = distribution.sample((n_samples,))

                data.append(samples)
                labels.append(torch.full((n_samples,), idx, dtype=torch.long, device=self.device))

        # Concatenate data and labels
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)

        # Shuffle the data
        permutation = torch.randperm(self.n_samples)
        data = data[permutation]
        labels = labels[permutation]

        return data, labels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
