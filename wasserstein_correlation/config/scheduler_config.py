from typing import Callable

class SchedulerConfig:
    def __init__(self, 
                 scheduler_class: Callable, 
                 params: dict):

        self.scheduler_class = scheduler_class
        self.params = params