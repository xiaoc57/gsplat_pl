# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.utils.data as data

from pathlib import Path
from PIL import Image

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

class GenerateOneImageDataset(data.Dataset):
    
    def __init__(self, img_path, height: int = 256, width: int = 256, **kwargs):
        super().__init__()
        if img_path:
            gt_image = image_path_to_tensor(img_path)
        else:
            gt_image = torch.ones((height, width, 3)) * 1.0
            # make top left and bottom right red, blue
            gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
            gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

        self.image = [gt_image]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return self.image[idx]
    
    
    
    # def check_files(self):
    #     # This part is the core code block for load your own dataset.
    #     # You can choose to scan a folder, or load a file list pickle
    #     # file, or any other formats. The only thing you need to gua-
    #     # rantee is the `self.path_list` must be given a valid value. 
    #     file_list_path = op.join(self.data_dir, 'file_list.pkl')
    #     with open(file_list_path, 'rb') as f:
    #         file_list = pkl.load(f)

    #     fl_train, fl_val = train_test_split(
    #         file_list, test_size=0.2, random_state=2333)
    #     self.path_list = fl_train if self.train else fl_val

    #     label_file = './data/ref/label_dict.pkl'
    #     with open(label_file, 'rb') as f:
    #         self.label_dict = pkl.load(f)