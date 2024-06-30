import os

from torch.utils.data import Dataset, DataLoader
from torchvision import io
from torchvision.tv_tensors import Mask, Image

from torch.utils import data

class SegmentationDataset(Dataset):
    "A class for image segmentation datasets."
    def __init__(self, root_dir):
        super().__init__()
        # get images
        imagetytpes = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.image_paths = [os.path.join(root, fname) for root, _, files in os.walk(self.image_dir) for fname in files if fname.endswith(imagetytpes)]
        self.label_paths = [os.path.join(root, fname) for root, _, files in os.walk(self.label_dir) for fname in files if fname.endswith(imagetytpes)]
        
        self.transform = None
        
        # Ensure images and labels are aligned
        self.image_paths.sort(key=lambda x: os.path.basename(x))
        self.label_paths.sort(key=lambda x: os.path.basename(x))
        
        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels must be equal"
        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            assert os.path.basename(img_path) == os.path.basename(lbl_path), f"Mismatch: {img_path} and {lbl_path}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
            
        img = Image(io.read_image(img_path))
        lbl = Mask(io.read_image(lbl_path))
        
        if self.transform:
            img, lbl = self.transform(img, lbl)
        
        # binarize
        lbl = lbl > 0
        lbl = lbl.float() # Cast back to float, since x is a ByteTensor now
        
        return img, lbl

class SegmentationInferenceDataset(Dataset):
    # Dataset for inference (no labels)
    def __init__(self, root, is_stereo, transform=None):
        super().__init__()
        # scan for images in the root directory
        self.root = root
        self.is_stereo = is_stereo
        
        # scan for images in the root
        self.image_paths = []
        for root, _, files in os.walk(self.root):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))
        
        self.transform = transform
        
    def __len__(self):
        # factor 2 if each image is stereo image
        return (2 if self.is_stereo else 1) * len(self.image_paths)

    '''
    def __getitem__(self, idx):
        img_path = self.image_paths(idx)
        img = Image(io.read_image(img_path))
        
        # split stereo image
        _, _, w = img.shape
        mid_point = w // 2
        
        left_img = img[:,:,:mid_point]
        right_img = img[:,:,mid_point:]
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        return left_img, right_img
    '''
    def __getitem__(self, idx):
        stereo_idx = idx // (2 if self.is_stereo else 1)  # Index of the stereo image, if stereo mode
        is_right = idx % 2     # Determine if it is left or right image

        img_path = self.image_paths[stereo_idx]
        
        # Use torchvision to load the image
        img = io.read_image(img_path).float() / 255.0  # Normalize to [0, 1]

        # Calculate the middle point to split the stereo image
        _, _, width = img.shape
        mid_point = width // 2
        
        # if stereo, cut image in half
        if self.is_stereo:
            if is_right:
                img = img[:, :, mid_point:]
            else:
                img = img[:, :, :mid_point]
            
        if self.transform:
            img = self.transform(img)
        
        return img
    
    def __get_img_name__(self,idx):
        # compute stereo idx if stereo mode
        stereo_idx = idx // (2 if self.is_stereo else 1)
        img_path = self.image_paths[stereo_idx]
        return os.path.basename(img_path)

def build_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size = 6):
    n_cpu = os.cpu_count()
    
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=n_cpu)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)#, num_workers=n_cpu)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)#, num_workers=n_cpu)
    
    return train_dl, valid_dl, test_dl
