import os

from torch.utils.data import Dataset, DataLoader
from torchvision import io
from torchvision.tv_tensors import Mask, Image


class SegmentationDataset(Dataset):
    "A class for image segmentation datasets."
    def __init__(self, root_dir):
        super().__init__()
        
        # get images
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.image_paths = [os.path.join(self.image_dir, fname) for fname in os.listdir(self.image_dir) if fname.endswith('.png')]
        self.label_paths = [os.path.join(self.label_dir, fname) for fname in os.listdir(self.label_dir) if fname.endswith('.png')]
        
        self.transform = None
        
        # Ensure images and labels are aligned
        self.image_paths.sort()
        self.label_paths.sort()
        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels must be equal"
        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            assert os.path.basename(img_path) == os.path.basename(lbl_path), f"Mismatch: {img_path} and {lbl_path}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
            
        img = Image(io.read_image(img_path))
        lbl = Mask(io.read_image(lbl_path), )
        
        if self.transform:
            img, lbl = self.transform(img, lbl)
        
        # binarize
        lbl = lbl > 0
        lbl = lbl.float() # Cast back to float, since x is a ByteTensor now
        
        return img, lbl

def build_dataloaders(train_dataset, valid_dataset, test_dataset):
    n_cpu = os.cpu_count()
    
    train_dl = DataLoader(train_dataset, batch_size=6, shuffle=True)#, num_workers=n_cpu)
    valid_dl = DataLoader(valid_dataset, batch_size=6, shuffle=False)#, num_workers=n_cpu)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)#, num_workers=n_cpu)
    
    return train_dl, valid_dl, test_dl
