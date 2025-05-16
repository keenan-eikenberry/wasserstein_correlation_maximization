import torch

class ModelLoader:
    def __init__(self, checkpoint_path: str, encoder, decoder=None):
        """
        Args:
            checkpoint_path: Full path to checkpoint file
        """
        self.checkpoint_path = checkpoint_path
        self.encoder = encoder 
        self.decoder = decoder
        
        self.load_checkpoint()
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.encoder.eval() 
        
        if self.decoder is not None and 'decoder' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.decoder.eval()
            
        return self.encoder, self.decoder

    def to(self, device):
        """Move models to specified device"""
        self.encoder.to(device)
        if self.decoder is not None:
            self.decoder.to(device)

        return self