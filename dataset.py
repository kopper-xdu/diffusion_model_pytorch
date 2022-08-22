from torchvision.datasets import ImageFolder
import torch
from PIL import Image


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = ImageFolder(root).imgs[:50000]
        for i in range(50000):
            self.imgs[i] = self.imgs[i][0]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
