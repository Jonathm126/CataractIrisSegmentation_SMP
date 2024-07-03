import json
import os
import pandas as pd
from IPython.display import display as ipy_display
from torchvision.transforms import v2 as T

def load_configs(config_name, paths_path):
    "Load Config Files"
    try:
        paths_path = os.path.join('configs/paths', paths_path)
        with open(paths_path, 'r') as f:
            paths = json.load(f)
    except: raise ValueError("no pths json :(")
    
    try:
        with open(os.path.join('configs/',config_name)) as f:
            config = json.load(f)
    except: raise ValueError("no config file :(")
    
    # update log path for this model
    paths['log_path'] = os.path.join(paths['log_path'], config['NAME'])
    paths['inference_save_path'] = os.path.join(paths['inference_save_path'],config['NAME'])
    
    # torch setup
    os.environ['TORCH_HOME'] = 'models/torch'
    
    # transforms build
    transforms_config = config.get("TRANSFORMS", {})
    data_transforms = {}
    # build of each phase
    for phase, phase_transforms in transforms_config.items():
        transform_list = []
        for transform_name, params in phase_transforms.items():
            transform_class = getattr(T, transform_name)
            if params:
                transform_list.append(transform_class(**params))
            else:
                transform_list.append(transform_class())
        # display
        
        data_transforms[phase] = T.Compose(transform_list)
        
    # print the configs
    df = pd.DataFrame(list(config.items()), columns=['Key', 'Value'])
    ipy_display(df)
    # print the transforms
    for phase, transforms in config['TRANSFORMS'].items():
        df_transforms = pd.DataFrame(transforms.items())
        ipy_display(df_transforms)
    
    return paths, config, data_transforms
