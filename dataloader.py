import matplotlib.image as mpimg
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
import torch.nn.functional as F
import glob
import natsort

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DataLoaderSegmentation(Dataset):
    def __init__(self, path):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(path,'input_image','*.jpg')) 
        self.mask_files = glob.glob(os.path.join(path,'hair_mask','*.png')) 
        
        self.img_files = natsort.natsorted(self.img_files)
        self.mask_files = natsort.natsorted(self.mask_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        
        
        img = mpimg.imread(img_path)/255
        img = torch.from_numpy(img).float()
        
        mask = mpimg.imread(mask_path)
        mask = torch.from_numpy(mask).float()
        
        return img, mask
    
    def __len__(self):
        return len(self.img_files)
    
# Load data
dataset = DataLoaderSegmentation("dataset")