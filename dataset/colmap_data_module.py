
import os
import torch
import torch.utils.data as data

from pathlib import Path
from PIL import Image

import lightning as L
from torch.utils.data import DataLoader, Dataset

from .camera import Camera
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
    
class ColmapDataModule(L.LightningDataModule):
    
    def __init__(self, path) -> None:
        super().__init__()
        

# 读入图像和相机，相机包含两个部分，相机内参与相机外参，相机内参是相同的，读入一次即可

class ColmapDataset(Dataset):
    
    def __init__(self, path: str, train: bool = True):
        super().__init__()

        self.path = path
        self.data = []
        self.intrisics = self.read_intrinsics()
    
    # read cameras' intrinsic, all the intrinsics will be the same. 
    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.path, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height)
        w = int(camdata[1].width)
        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]
            cx = camdata[1].params[1]
            cy = camdata[1].params[2]
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]
            fy = camdata[1].params[1]
            cx = camdata[1].params[2]
            cy = camdata[1].params[3]
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "height": h, "width": w,
        }
    
    def read_image
    
    def read_meta(self):
        
        '''
            read the metadata.
            
        '''
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        

if __name__ == "__main__":
    pass