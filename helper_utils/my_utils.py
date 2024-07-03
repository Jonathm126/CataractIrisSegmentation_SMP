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
    
    # print the configs
    df = pd.DataFrame(list(config.items()), columns=['Key', 'Value'])
    ipy_display(df)
    
    return paths, config

def print_transforms(data_transforms):
    data = []
    for transform_name, transform in data_transforms.items():
        for t in transform.transforms:
            data.append({"Set": transform_name, "Transform": t.__class__.__name__, "Details": str(t)})

    df = pd.DataFrame(data).groupby('Set')
    for name, group in df:
        print(f"Transforms for {name} set:")
        pd.set_option('display.max_colwidth', None)
        ipy_display(group.reset_index(drop=True))
        print("\n")
    ipy_display(df)
