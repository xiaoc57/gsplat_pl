from typing import Dict, List, Literal, Optional, Tuple, Union
from jaxtyping import Float, Int, Shaped
from torch import Tensor

CAMERA_MODEL_TO_TYPE = {
    "SIMPLE_PINHOLE": "CameraType.PERSPECTIVE", # 只实现简单地针孔相机模型
}


# 这里只创建一个相机的模型，不是相机组，如果要读入的时候就是一个一个的
class Camera:
    
    def __init__(self, 
                 camera_to_world: Float[Tensor, "3 4"],
                 fx: float,
                 fy: float,
                 cx: float, 
                 cy: float,
                 width: int,
                 height: int
                 ):
        
        # camera_to_worlds: Float[Tensor, "3 4"],
        # fx: Union[Float[Tensor, "*batch_fxs 1"], float],
        # fy: Union[Float[Tensor, "*batch_fys 1"], float],
        # cx: Union[Float[Tensor, "*batch_cxs 1"], float],
        # cy: Union[Float[Tensor, "*batch_cys 1"], float],
        # width: Optional[Union[Shaped[Tensor, "*batch_ws 1"], int]] = None,
        # height: Optional[Union[Shaped[Tensor, "*batch_hs 1"], int]] = None,
        
        self.camera_to_world = camera_to_world
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        
        
        
        
        
        
        
        
        
        
        