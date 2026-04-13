import torch
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize


class InfiniteSampler(Sampler):
    """Sampler that yields indices infinitely, reshuffling each epoch."""

    def __init__(self, data_source):
        super().__init__()
        self.N = len(data_source)

    def __iter__(self):
        while True:
            for idx in torch.randperm(self.N):
                yield idx


class PathologyLoader:
    """Data loader for multi-domain pathology images using ImageFolder format.

    Expected directory structure:
        data_dir/
            domain_0/
                image_0.jpg
            domain_1/
                image_1.jpg
            ...
    """

    def __init__(self, opt):
        self.opt = opt
        self.dir_path = opt.dir_path
        self.transform = self._transform()

        self.dataset = self.load_data()
        self.num_classes = len(self.dataset.classes)
        self.dataloader = self.make_loader(self.dataset)
        self.dataloader_iter = self.make_iter(self.dataset)

    def load_data(self):
        return ImageFolder(self.dir_path, transform=self.transform)

    def make_loader(self, dataset):
        return DataLoader(dataset, self.opt.batch_size)

    def make_iter(self, dataset):
        return iter(DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            num_workers=0,
            sampler=InfiniteSampler(dataset),
        ))

    def _transform(self):
        return transforms.Compose([
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(self.opt.img_shape, (0.05, 1)),
        ])


class TestLoader(PathologyLoader):
    """Data loader for test images (no augmentation)."""

    def __init__(self, opt):
        self.opt = opt
        self.dir_path = opt.dir_test_path
        self.transform = self._transform()

        self.dataset = self.load_data()
        self.num_classes = len(self.dataset.classes)
        self.dataloader = self.make_loader(self.dataset)
        self.dataloader_iter = self.make_iter(self.dataset)

    def _transform(self):
        return transforms.Compose([
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Resize(size=self.opt.img_shape),
        ])
