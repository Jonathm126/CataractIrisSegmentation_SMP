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

class Run_params:
    def __init__(self):
        self.config_name = None
        self.paths_name = 'paths_pc.json'
        self.test_images = 3
        self.plot_metrics = ['dataset_iou', 'dataset_loss']
        
        self.run_on_validation = False
        self.time_validation = False
        self.print_validation_images = 10
        self.save_validation = False
        
        self.run_inference = False
        self.time_inference = False
        self.print_inference_images = 10
        self.save_inference = False

def main(run_params: Run_params):
    # setup
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch setup
    os.environ['TORCH_HOME'] = 'models/torch'

    # Load Configuration:
    config_name = run_params.config_name
    paths_name = run_params.paths_name
    paths, config = my_utils.load_configs(config_name, paths_name) # select paths file

    # load transforms - select v1 or v0
    from models.transforms import data_transforms_v1 as data_transforms 
    # display the transforms for debugging
    my_utils.print_transforms(data_transforms)

    # Data loading
    train_dataset = SegmentationDataset(os.path.join(paths['train_data_root'], 'train'), data_transforms['train'])
    valid_dataset = SegmentationDataset(os.path.join(paths['train_data_root'], 'valid'), data_transforms['valid'])
    train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, drop_last=True)#, num_workers=24)
    valid_dl = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True)#, num_workers=n_cpu)

    # Plot a few images to test
    for _ in range(run_params.test_images):
        index = random.randint(0, len(train_dataset) - 1)
        display_utils.display_sample(train_dataset, index)

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

    # Train and plot convergance:
    if config['MODE'] == 'train':
        # define here which metrics to plot after training
        plot_metrics = run_params.plot_metrics
        
        # train using the included logger
        trainer.fit(model, 
            train_dataloaders=train_dl, 
            val_dataloaders=valid_dl)
        
        # plot metrics after training
        if plot_metrics != None:
            pth = os.path.join(train_logger.log_dir, 'metrics.csv')
            display_utils.plot_losses(pth, plot_metrics)

    # ### Test on Validation data
    if run_params.run_on_validation:
#        model.eval()
#        with torch.no_grad():
#            trainer.test(model= model, dataloaders = valid_dl)
#
#        # display metrics on validation data
#        pth = os.path.join(paths['log_path'], 'test', 'metrics.csv')
#        display_utils.plot_losses(pth, ['image_iou', 'image_loss'], test = True)
        
        # run the model on validation data, to generate images 
        time = run_params.time_validation
        save_pth = os.path.join(paths['train_data_root'], 'infer', config['NAME'])
        timings = infer_set(model, device, save_pth, dataset=valid_dl.dataset, save=run_params.save_validation, 
                            to_print=run_params.print_validation_images, time = time, all = True)

        # save timings
        if time:
            output_file = os.path.join(test_csv_logger.log_dir,'timing.csv')
            os.makedirs(test_csv_logger.log_dir, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write('valid_dataset_v2'+'\n')
                timings.tofile(f, sep='\n', format='%f')

    ### Out-of-sample Inference
    if run_params.run_inference:
    # build dataset
        infer_dataset = SegmentationInferenceDataset(paths['inference_data_root'], transform=data_transforms['infer'], is_stereo=True)

        # run model on images
        time = run_params.time_inference
        save_pth = os.path.join(paths['inference_save_path'])
        timings = infer_set(model, device, save_pth, dataset=infer_dataset, save=run_params.save_inference, 
                            to_print=run_params.print_inference_images, time = time, all=True)

        # save timings
        if time:
            output_file = os.path.join(test_csv_logger.log_dir,'timing.csv')
            os.makedirs(test_csv_logger.log_dir, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write('inference_dataset'+'\n')
                timings.tofile(f, sep='\n', format='%f')


if __name__ == "__main__":
    run_params = Run_params()

    # CONFIGURE HERE
    run_params.config_name = "Unet_efficientnet-b3_load.json" # Name of model configuration file to load
    run_params.paths_name = 'paths_drive.json' # Name of paths configuration file
    
    # Training
    run_params.test_images = 0 # How many test images to print to make sure data was loaded
    run_params.plot_metrics = ['dataset_iou', 'dataset_loss'] # Which params to plot after training
    
    # Validation
    run_params.run_on_validation = True # Generate masks from validation dataset?
    run_params.time_validation = False # Record time?
    run_params.print_validation_images = 10 # How many validation images to display
    run_params.save_validation = False # Save images to drive?
    
    # Inference (On new images)
    run_params.run_inference = False # Generate masks from validation dataset?
    run_params.time_inference = False # Record time?
    run_params.print_inference_images = 10 # How many validation images to display
    run_params.save_inference = False # Save images to drive?
    
    main(run_params)
