class LossConfig:
    """
    Configuration for loss.
    """

    def __init__(self, 
                 reconstruct: float,
                 wasserstein_correlation: float):
        self.reconstruct = reconstruct
        self.wasserstein_correlation = wasserstein_correlation
   