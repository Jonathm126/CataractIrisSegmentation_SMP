# torch lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# utils
import os

def infer_on_image(model, image_path, transform):
    """
    Function to perform inference on a single image.
    
    Args:
    - model (nn.Module): The trained segmentation model.
    - image_path (str): Path to the input image.
    - transform (callable): Transformation to be applied to the image.
    
    Returns:
    - The inferred mask for the input image.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # Post-process the output
    inferred_mask = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    inferred_mask = (inferred_mask > 0.5).astype(np.uint8)  # Threshold the output
    
    return inferred_mask

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