from typing import Callable

class OptimizerConfig:
    def __init__(self, 
                 optimizer_class: Callable, 
                 params: dict):

        self.optimizer_class = optimizer_class
        self.params = params