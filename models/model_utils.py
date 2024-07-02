# torch lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, prediction_writer
from pytorch_lightning.loggers import CSVLogger
from torch import no_grad
from helper_utils import display
import random

# utils
import os

def build_trainer(config, paths):
    "Sets up the trainer for PL training of the SMP model"
    
    # Set up the logger 
    csv_logger = CSVLogger(paths['log_path'],name=None)

    # Create a ModelCheckpoint callback to save only the best model
    callback_path = os.path.join(csv_logger.save_dir, csv_logger.name, f'version_{csv_logger.version}')

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_dataset_loss',  # the metric to monitor
        dirpath=callback_path,
        filename=config['NAME']+'_best-checkpoint',  # name of the best model file
        save_top_k=1,  # save only the best model
        mode='min'  # save the model with minimum validation loss
    )

    # create a trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config.get('NUM_EPOCHS', 10),
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],  
        logger = csv_logger
    )
    return trainer, csv_logger

def infer_set(model, device, pth, dataset, save=True, print=None):
    # setup
    model.eval()
    model.to(device)
    
    # inference loop
    with no_grad():
        if save:
            for idx in range(len(dataset)):
                data = dataset[idx]
                # detect if we have a img, mask or just image
                if isinstance(data, tuple) and len(data) == 2: # we get a GT mask
                    img, gt_mask = data
                    # cuda
                    img = img.to(device)
                    # infer
                    pred_mask = model.infer(img).squeeze(0).squeeze(0).cpu()
                else:
                    img = data
                    pred_mask = None
                    # cuda
                    img = img.to(device)
                    # infer
                    gt_mask = model.infer(img).squeeze(0).squeeze(0).cpu()
                # do we have a loader that gives names? if so, use that
                img_name = f'test_{dataset.__get_img_name__(idx)}' if hasattr(dataset, '__get_img_name__') else f'test_{idx}.png'
                # save mask
                os.makedirs(pth, exist_ok=True)
                display.save_mask(img, mask1=gt_mask, mask2=pred_mask, path=os.path.join(pth, img_name))
        
        # print a few samples
        if print is not None:
            for idx in random.sample(range(len(dataset)), print):
                data = dataset[idx]
                # detect if we have a img, mask or just image
                if isinstance(data, tuple) and len(data) == 2: # we get a GT mask
                    img, gt_mask = data
                    pred_mask = model.infer(img.to(device)).squeeze(0).squeeze(0).cpu()
                else:
                    img = data
                    pred_mask = None
                    gt_mask = model.infer(img.to(device)).squeeze(0).squeeze(0).cpu()
                # infer mask
                # display
                display.display_mask(img, mask1=gt_mask, mask2=pred_mask, mode='overlay')
    