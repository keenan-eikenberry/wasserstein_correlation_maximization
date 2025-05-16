import torch
import torch.nn as nn

class PCA(nn.Module):
    def __init__(self, 
                 data,
                 n_components):
        super(PCA, self).__init__()
        self.data = data
        self.n_components = n_components
        self.device = data.device
        self.fit_pca(self.data)
        
    def fit_pca(self, data):
        """
        Fit PCA using SVD decomposition.
        """
        
        # Center data
        self.mean = torch.mean(data, dim=0, keepdim=True)
        data_centered = data - self.mean
        
        # SVD: X = U Σ W^T
        U, S, W_T = torch.linalg.svd(data_centered, full_matrices=False)
        W = W_T.T
        self.projection_matrix = W[:, :self.n_components]

        self.mean = self.mean.to(self.device)
        self.projection_matrix = self.projection_matrix.to(self.device)
        
    def forward(self, X):
        """
        Project data onto the principal components.
        """
        input_device = X.device
        if input_device != self.device:
            self.device = input_device
            self.mean = self.mean.to(input_device)
            self.projection_matrix = self.projection_matrix.to(input_device)
            
        if X.dim() > 2:
            X = X.flatten(1)

        single_sample = X.dim() == 1
      
        if single_sample:
            X = X.unsqueeze(0)

        # Center the data
        X_centered = X - self.mean
        
        # Project the data: X_transformed = X_centered @ W
        X_proj = X_centered @ self.projection_matrix
        
        if single_sample:
            X_proj = X_proj.squeeze(0)
        
        return X_proj
