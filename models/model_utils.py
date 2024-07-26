# torch lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch
from helper_utils import display_utils
import numpy as np

# utils
import os

def build_trainer(config, paths):
    "Sets up the trainer for PL training of the SMP model"
    
    # Set up the logger 
    csv_logger = CSVLogger(paths['log_path'],name=None)

    # Create a ModelCheckpoint callback to save only the best model
    callback_path = os.path.join(paths['checkpoint_path'], config['NAME'], f'version_{csv_logger.version}')

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_dataset_loss',  # the metric to monitor
        dirpath=callback_path,
        filename=config['NAME']+'_best-checkpoint',  # name of the best model file
        save_top_k=1,  # save only the best model
        mode='min'  # save the model with minimum validation loss
    )
    
    # Create an EarlyStopping callback to stop training when the monitored metric stops improving
    early_stopping_callback = EarlyStopping(
        monitor='valid_dataset_loss',  # the metric to monitor
        min_delta=0.001,  # minimum change in the monitored metric to qualify as improvement
        patience=5,  # number of epochs with no improvement after which training will be stopped
        mode='min'  # mode of the monitored metric (minimize loss)
    )

    # create a trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config.get('NUM_EPOCHS', 10),
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stopping_callback],  
        logger = csv_logger
    )
    return trainer, csv_logger

def infer_set(model, device, pth, dataset, save=False, to_print=None, time=False, all = False):
    # setup
    model.eval()
    model.to(device)

    # decide on what to iterate
    print_idx = np.random.randint(low=0, high=len(dataset), size=to_print)
    itr_range = range(len(dataset)) if (save or all) else print_idx
    
    # prep for time measurement
    if time:
        # cuda timing
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # warmup
        dummy_input = torch.randn(1,3,640,640, dtype=torch.float).to(device)
        for _ in range(10):
            _ = model(dummy_input)
        timings = []
    
    # inference loop
    with torch.no_grad():
        # infer and measure time
        for idx in itr_range:
            # get data
            data = dataset[idx]
            if isinstance(data, tuple) and len(data) == 2: # we get a GT mask
                img, gt_mask = data
            else: # only image
                img = data
                gt_mask = None
            # cuda
            img = img.to(device)
            # time if needed
            if time:
                starter.record()
            # predict
            pred_mask, _ = model.infer(img)
            pred_mask = pred_mask.squeeze(0).squeeze(0)
            # time after prediction
            if time:
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)
            # move to cpu for post processing
            pred_mask = pred_mask.cpu()
            # save if needed
            if save:
                # do we have a loader that gives names? if so, use that
                img_name = f'test_{dataset.__get_img_name__(idx)}' if hasattr(dataset, '__get_img_name__') else f'test_{idx}.png'
                # save mask
                os.makedirs(pth, exist_ok=True)
                display_utils.save_mask(img, mask1=gt_mask, mask2=pred_mask, path=os.path.join(pth, img_name))

            # print a few samples if needed
            if to_print is not None:
                if idx in print_idx:
                    # check if this is an image to print
                    display_utils.display_mask(img, mask1=gt_mask, mask2=pred_mask, mode='overlay')
        
        return np.array(timings) if time else None