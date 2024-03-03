
import lightning as L

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 


class SimpleData(L.LightningDataModule):

    def __init__(self, root_dir, batch_size, num_workers=8, **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.kwargs = kwargs
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 下载训练集 MNIST 手写数字训练集
        train_dataset = datasets.MNIST(
            root=self.root_dir, train=True, transform=transforms.ToTensor(), download=True)
        
        test_dataset = datasets.MNIST(
            root=self.root_dir, train=False, transform=transforms.ToTensor())

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = train_dataset
            self.valset = test_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
