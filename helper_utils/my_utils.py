import json
import os

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
    paths['inference_save_path'] = os.path.join(paths['inference_data_root'], paths['inference_save_path'],config['NAME'])

    # print config
    print(json.dumps(config, indent=2))
    
    # torch setup
    os.environ['TORCH_HOME'] = 'models/torch'
    
    return paths, config
