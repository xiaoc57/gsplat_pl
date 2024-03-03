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


class SimpleSplat(L.LightningModule):
    
    def __init__(self, num_points: int = 2000, kwargs = None):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.frames = []
        self.kwargs = kwargs
        self.num_points = num_points
        
        fov_x = math.pi / 2.0
        self.H, self.W = kwargs.get("h"), kwargs.get("w")
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1])
        self._init_gaussians()

    def setup(self, stage):
        
        if self.kwargs and self.kwargs.get("learning_rate"):
            self.learning_rate = self.kwargs.get("learning_rate")
        else:
            self.learning_rate = 0.02
        
        
    
    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2

        self.means = nn.Parameter(bd * (torch.rand(self.num_points, 3) - 0.5))
        self.scales = nn.Parameter(torch.rand(self.num_points, 3))
        d = 3
        self.rgbs = nn.Parameter(torch.rand(self.num_points, d))

        u = torch.rand(self.num_points, 1)
        v = torch.rand(self.num_points, 1)
        w = torch.rand(self.num_points, 1)

        
        quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.quats = nn.Parameter(quats)
        self.opacities = nn.Parameter(torch.ones((self.num_points, 1)))
        
        viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        self.viewmat = nn.Parameter(viewmat)
        
        self.background = nn.Parameter(torch.zeros(d))
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False
        

    def training_step(self, batch, batch_nb: int):
        B_SIZE = 16
        (
            xys,
            depths,
            radii,
            conics,
            compensation,
            num_tiles_hit,
            cov3d,
        ) = project_gaussians(
            self.means,
            self.scales,
            1,
            self.quats,
            self.viewmat,
            self.viewmat,
            self.focal,
            self.focal,
            self.W / 2,
            self.H / 2,
            self.H,
            self.W,
            B_SIZE,
        )
        # torch.cuda.synchronize()
        
        out_img = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            torch.sigmoid(self.rgbs),
            torch.sigmoid(self.opacities),
            self.H,
            self.W,
            B_SIZE,
            self.background,
        )[..., :3]
        # torch.cuda.synchronize()
        loss = self.mse_loss(out_img, batch[0])
        
        save_imgs = self.kwargs.get("save_imgs")
        
        if save_imgs and self.current_epoch % 5 == 0:
            self.frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        
        return loss
    
    def on_train_end(self):
        save_imgs = self.kwargs.get("save_imgs")
        if save_imgs:
            # save them as a gif with PIL
            self.frames = [Image.fromarray(frame) for frame in self.frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            self.frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=self.frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)
    
    
        
        
        
        
        