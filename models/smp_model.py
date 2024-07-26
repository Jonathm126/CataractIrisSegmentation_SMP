# torch
import torch
import torch.utils
import torch.utils.benchmark
import torch.utils.data
import pytorch_lightning as pl

# segmentation models
import segmentation_models_pytorch as smp

# loss functions
from models.dice_edge_loss import contour_dice_loss

# custom utils
from helper_utils.display_utils import display_for_epoch

class CatSegModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        # data
        self.config = config
        # save hparams only if training
        self.save_hyperparameters(logger=self.training)

        # per - epoch logs
        self.train_outputs = []
        self.valid_outputs = []
        self.test_outputs = []
        
        # specify default params
        num_classes = len(config['CLASSES'])
        encoder_weights = config.get('ENCODER_WEIGHTS', None)
        
        # build the model using SMP
        try:
            self.model = smp.create_model(
                arch=config['ARCH'], 
                encoder_name=config['ENCODER'], encoder_weights=encoder_weights,
                in_channels=config['IN_CH'], classes=num_classes,
                **kwargs
            )
        except: raise ValueError(f"Error creating model with given configuration.")
        
        # preprocessing parameters for image
        # we get the same preprocessing params used in the training of the encoder
        params = smp.encoders.get_preprocessing_params(config['ENCODER'])
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(config['ENCODER'], encoder_weights)
        
        # get loss function
        # for image segmentation dice loss could be the best first choice unless other specified
        try:
            loss_fn_name = config.get('LOSS', 'DiceLoss')
            if hasattr(smp.losses, loss_fn_name):
                self.loss_fn = getattr(smp.losses, loss_fn_name)(smp.losses.BINARY_MODE)
            elif loss_fn_name == 'CountourDiceLoss':
                self.loss_fn = contour_dice_loss
            else:raise ValueError(f"Unknown loss function defined.")
        except: raise ValueError(f"Cannot define loss function.")

    def forward(self, image):
        
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask
    
    def infer(self,image, time=False):
        # assertion
        h,w = image.shape[-2:]
        assert h%32==0 and w%32==0
                
        # cuda timing
        if time:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
        
        #with torch.no_grad():
            # do a step
        logits_mask = self(image)
        
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        # timing
        if time:
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
        
        else:
            curr_time = None
        
        return pred_mask, curr_time

    def shared_step(self, batch, stage):
        image, mask = batch
        
        # assertion
        h,w = image.shape[2:]
        assert h%32==0 and w%32==0
        assert image.ndim==4
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        # do a step
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        # compute metrics
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        
    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        dataset_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_dataset_loss": dataset_loss
        }
        if "stage" != 'test':
            self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False, logger=True)

    def training_step(self, batch, batch_idx):
        result = self.shared_step(batch, "train")
        self.train_outputs.append(result)
        return result

    def validation_step(self, batch, batch_idx):
        result = self.shared_step(batch, "valid")
        self.valid_outputs.append(result)
        return result

    def test_step(self, batch, batch_idx):
        result = self.shared_step(batch, "test")
        self.test_outputs.append(result)
        
        tp = result['tp'].to(self.device)
        fp = result['fp'].to(self.device)
        fn = result['fn'].to(self.device)
        tn = result['tn'].to(self.device)
                
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn)
        
        image_metrics = {
            "test_image_loss": result["loss"].item(),
            "test_image_iou": per_image_iou.item()
        }
        
        self.log_dict(image_metrics, logger=True, on_epoch=False, on_step=True)
        
        return result

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.train_outputs, "train")
        self.train_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.valid_outputs, "valid")
        self.valid_outputs.clear()
        display_for_epoch(self, self.trainer.val_dataloaders.dataset, sample_indices=[0], device=self.device)

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_outputs, "test")
        self.test_outputs.clear()

    def configure_optimizers(self):
        lr = self.config.get('LR', 0.0001)
        return torch.optim.Adam(self.parameters(), lr=lr)
    