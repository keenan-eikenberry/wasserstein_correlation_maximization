from torch import nn
from typing import Optional, Tuple

class PatchEncoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 output_shape: Tuple[int, ...],
                 local_encoder: nn.Module, 
                 latent_mixing_network: Optional[nn.Module], 
                 patch_h: int, 
                 patch_w: int, 
                 stride_h: int, 
                 stride_w: int):
        super(PatchEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.channels = input_shape[0]
        self.output_shape = output_shape
        self.local_encoder = local_encoder 
        self.latent_mixing_network = latent_mixing_network
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.stride_h = stride_h
        self.stride_w = stride_w 

        # Height and width for grid of data patches 
        self.grid_h, self.grid_w = self._compute_grid_size()
        self.patch_input_shape = (self.channels, self.patch_h, self.patch_w)

    
    def _compute_grid_size(self): 
        grid_h = (self.input_shape[1] - self.patch_h) // self.stride_h + 1
        grid_w = (self.input_shape[2] - self.patch_w) // self.stride_w + 1

        return grid_h, grid_w

    def _patchify(self, data):
        # Input shape (batch_size, channels, height, width)
        # Output shape (batch_size, grid_h, grid_w, channels, patch_h, patch_w) 
        # where grid_h = (height - patch_h) // stride_h + 1
        # and grid_w = (width - patch_w) // stride_w + 1

        if data.dim() == 3: 
            data = data.unsqueeze(1) 
        
        batch_size = data.shape[0]
        patches = data.unfold(2, self.patch_h, self.stride_h).unfold(3, self.patch_w, self.stride_w)
        patches = patches.contiguous().view(batch_size, self.channels, self.grid_h, self.grid_w, self.patch_h, self.patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        return patches
            
    def forward(self, x):
        batch_size = x.shape[0]
        x_patch = self._patchify(x)
        data_patches = x_patch
        x_patch = x_patch.flatten(start_dim=1, end_dim=2)
        x_patch = x_patch.flatten(start_dim=0, end_dim=1)
        latent_patches = self.local_encoder(x_patch)
        latent_patches = latent_patches.unflatten(0, (batch_size, self.grid_h, self.grid_w))

        # (batch, grid_h, grid_w, channels, patch_h, patch_w)
        if latent_patches.dim() == 6: 
            latent_code = latent_patches.flatten(start_dim=3, end_dim=5)
            latent_code = latent_code.flatten(start_dim=1, end_dim=2)
            latent_code = latent_code.flatten(start_dim=1, end_dim=2)
        # (batch, grid_h, grid_w, latent_dim)
        elif latent_patches.dim() == 4: 
            latent_code = latent_patches.flatten(start_dim=1, end_dim=2)
            latent_code = latent_code.flatten(start_dim=1, end_dim=2)

        if self.latent_mixing_network is not None:
            latent_code = self.latent_mixing_network(latent_code)

        return data_patches, latent_patches, latent_code