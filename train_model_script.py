# %%
%load_ext autoreload
%autoreload 2

# %%
# torch
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger

# utils
import random

# custom utils
from datasets.dataset import *
from helper_utils import display_utils, my_utils
from models.smp_model import CatSegModel
from models.model_utils import build_trainer, infer_set

# %%
# setup
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
# torch setup
os.environ['TORCH_HOME'] = 'models/torch'

# %% [markdown]
# #### Configuration
# Load Configuration:

# %%
config_name = "Unet_resnet34_load.json"
paths, config = my_utils.load_configs(config_name, 'paths_pc.json') # select paths file
from models.transforms import data_transforms_v1 as data_transforms # select the correct transform!

# %% [markdown]
# Print Transforms:

# %%
my_utils.print_transforms(data_transforms)

# %% [markdown]
# ### Data prep

# %% [markdown]
# Build dataset:

# %%
# get files
train_dataset = SegmentationDataset(os.path.join(paths['train_data_root'], 'train'), data_transforms['train'])
valid_dataset = SegmentationDataset(os.path.join(paths['train_data_root'], 'valid'), data_transforms['valid'])

# dataloaders
train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, drop_last=True)#, num_workers=24)
valid_dl = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True)#, num_workers=n_cpu)

# %% [markdown]
# Plot a few images for example:

# %%
for _ in range(3):
    index = random.randint(0, len(train_dataset) - 1)
    display_utils.display_sample(train_dataset, index)

# %% [markdown]
# ### Build model
# Config load:

# %%
# build model
if config['MODE'] == 'train':
    model = CatSegModel(config)
    
elif config['MODE'] == 'load':
    # load model from checkpoint
    model = CatSegModel.load_from_checkpoint(
        checkpoint_path=os.path.join(paths['checkpoint_path'], config['NAME'],config['NAME']+'_best-checkpoint.ckpt'),
        #hparams_file=os.path.join(paths['log_path'], 'hparams.yaml')
    )

# Build the PL trainer:
trainer, train_logger = build_trainer(config, paths)

test_csv_logger = CSVLogger(paths['log_path'], 'test')
trainer.logger = test_csv_logger

# %% [markdown]
# #### Train and plot convergance:

# %%
plot_metrics = ['dataset_iou', 'dataset_loss']

if config['MODE'] == 'train':
    # train using the included logger
    trainer.fit(model, 
        train_dataloaders=train_dl, 
        val_dataloaders=valid_dl)
    
    if plot_metrics != None:
        pth = os.path.join(train_logger.log_dir, 'metrics.csv')
        display_utils.plot_losses(pth, plot_metrics)

# %% [markdown]
# ### Test on Validation data

# %%
model.eval()
with torch.no_grad():
    trainer.test(model= model, dataloaders = valid_dl)

# %% [markdown]
# Plot validation preformance:

# %%
pth = os.path.join(paths['log_path'], 'test', 'metrics.csv')
display_utils.plot_losses(pth, ['image_iou', 'image_loss'], test = True)

# %% [markdown]
# ### Generate marked validation images
# We will begin by generating the predicted masks on the validation dataset, and save them to disk in `train_data_root\infer\NAME`.
# Also, we will display a few images.

# %%
time = True # should we time this

save_pth = os.path.join(paths['train_data_root'], 'infer', config['NAME'])
timings = infer_set(model, device, save_pth, dataset=valid_dl.dataset, save=False, to_print=None, time = time, all = True)

# save timings
if time:
    output_file = os.path.join(test_csv_logger.log_dir,'timing.csv')
    os.makedirs(test_csv_logger.log_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('valid_dataset_v2'+'\n')
        timings.tofile(f, sep='\n', format='%f')

# %% [markdown]
# ### Out-of-sample Inference
# For qualiative evaluation we will generate and display images from out-of-dataset.
# 
# First we need to load the inference dataset `infer_dataset`:

# %%
# build dataset
infer_dataset = SegmentationInferenceDataset(paths['inference_data_root'], transform=data_transforms['infer'], is_stereo=True)

# %% [markdown]
# Generate the masks:

# %%
time = False # select
save_pth = os.path.join(paths['inference_save_path'])
timings = infer_set(model, device, save_pth, dataset=infer_dataset, save=True, to_print=10, time = time, all=True)

# save timings
if time:
    output_file = os.path.join(test_csv_logger.log_dir,'timing.csv')
    os.makedirs(test_csv_logger.log_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('inference_dataset'+'\n')
        timings.tofile(f, sep='\n', format='%f')


