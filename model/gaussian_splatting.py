import os
import numpy as np
import torch
import math
import lightning as L
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from PIL import Image
from torch import Tensor, optim
import torch.nn as nn


class GaussianSplatting(L.LightningModule):
    
    '''
        init from sfm or not
        
    '''
    def __init__(self, *args: os.Any, **kwargs: os.Any) -> None:
        super().__init__(*args, **kwargs)
        
    def setup(self, stage):
        
        '''
            read ply or random
        '''
        
    
    
