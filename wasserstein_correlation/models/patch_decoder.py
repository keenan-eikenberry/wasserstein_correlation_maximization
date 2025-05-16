from torch import nn
from typing import Optional, Tuple

class PatchDecoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 output_shape: Tuple[int, int, int], 
                 latent_unmixing_network: Optional[nn.Module],
                 sub_latent_dim: Optional[int], 
                 local_decoder: nn.Module,
                 data_mixing_network: Optional[nn.Module], 
                 patch_h: int, 
                 patch_w: int, 
                 stride_h: int, 
                 stride_w: int):
        super(PatchDecoder, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.channels = output_shape[0]
        self.latent_unmixing_network = latent_unmixing_network
        self.sub_latent_dim = sub_latent_dim
        self.local_decoder = local_decoder 
        self.data_mixing_network = data_mixing_network
        self.patch_h = patch_h
        self.patch_w = patch_w 
        self.stride_h = stride_h
        self.stride_w  = stride_w
        self.grid_h, self.grid_w = self._compute_grid_size()
        self.patch_input_shape = (self.channels, self.patch_h, self.patch_w)

    def _compute_grid_size(self): 
        grid_h = (self.output_shape[1] - self.patch_h) // self.stride_h + 1
        grid_w = (self.output_shape[2] - self.patch_w) // self.stride_w + 1

        return grid_h, grid_w

    def forward(self, x):
        batch_size = x.shape[0]
        if self.latent_unmixing_network is not None:
            x = self.latent_unmixing_network(x)
            x = x.unflatten(1, (self.grid_h, self.grid_w, self.sub_latent_dim))
            latent_patches = x

        x = x.flatten(start_dim=1, end_dim=2)
        x = x.flatten(start_dim=0, end_dim=1)
        x = self.local_decoder(x)
        x = x.unflatten(0, (batch_size, self.grid_h, self.grid_w))
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        data_patches = x
        x = x.view(batch_size, self.channels, self.grid_h * self.patch_h, self.grid_w * self.patch_w)
        
        if self.data_mixing_network:
            x = self.data_mixing_network(x)
            x = x.reshape(-1, *self.output_shape)

        return x, data_patches, latent_patches 