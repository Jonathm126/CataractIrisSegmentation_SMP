from torchvision.transforms import v2 as T
from torchvision.tv_tensors import Image, Mask
from torch import uint8


class CustomPerspective(T.RandomPerspective):
    def __init__(self, distortion_scale=0.5, p=0.5, fill=None):
        super().__init__(distortion_scale=distortion_scale, p=p, fill=fill)
    
    def _transform(self, inpt, params):
        if isinstance(inpt, Image):
            mean_fill = inpt.float().mean().item() 
            self.fill[Image] = mean_fill
        
        return super()._transform(inpt, params)
    

data_transforms_v1 = {}
data_transforms_v1['train'] = T.Compose([
    T.ToImage(),
    T.RandomVerticalFlip(p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop(size=640, scale=(0.7, 1.1)),#, fill={Image: 127, Mask: 0}),
    CustomPerspective(distortion_scale = 0.2, p = 0.6, fill = {Image:[0, 0, 0], Mask: 0}),
    T.CenterCrop(640),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    #T.ToDtype(torch.float32, scale=True)
])

data_transforms_v1['valid'] = T.Compose([
    T.ToImage(),
    T.CenterCrop(640)
    #T.ToDtype(torch.float32, scale=True)
])

data_transforms_v1["infer"] = T.Compose([
    T.ToImage(),
    T.Resize(640-256, 640-32),
    T.Pad([16,128,16,128], fill=[0.5,0.5,0.5]),
    T.ToDtype(uint8, scale=True)
])