import torch
import yaml
from pprint import pprint
from collections import OrderedDict

def print_object_attributes(obj):
    pprint(vars(obj))

class OrderedDumper(yaml.Dumper):
    def represent_dict(self, data):
        return self.represent_mapping('tag:yaml.org,2002:map', data.items())

def str_representer(dumper, data):
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

OrderedDumper.add_representer(str, str_representer)
OrderedDumper.add_representer(OrderedDict, OrderedDumper.represent_dict)

# Save YAML Config
def save_yaml_config(config_dict, file_path):
    with open(file_path, 'w') as f:
        yaml.dump(config_dict, f, Dumper=OrderedDumper, default_flow_style=False)

def config_to_dict(config):
    # Special cases
    if callable(config) and (config.__module__.endswith('optim') or config.__module__.endswith('lr_scheduler')):
        return f"{config.__module__.split('.')[-1]}.{config.__name__}"
    elif isinstance(config, torch.utils.data.Sampler):
        return "sampler"
    elif isinstance(config, torch.Generator):
        return "torch.Generator()"
    
    # General cases
    elif isinstance(config, (list, tuple)):
        return [config_to_dict(v) for v in config]
    elif isinstance(config, dict):
        return OrderedDict({k: config_to_dict(v) for k, v in config.items() if not callable(v)})
    elif hasattr(config, "__dict__"):
        return OrderedDict({k: config_to_dict(v) 
                            for k, v in config.__dict__.items() 
                            if not callable(v) 
                            and not isinstance(v, (torch._C.Generator, torch.device, torch.Tensor))
                            and hasattr(config, k) 
                            })
    
    return config