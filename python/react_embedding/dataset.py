import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset


class InstancesDataset(Dataset):

    def __init__(self, data_root: str, transforms):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms

        self.labels = []
        self.images = []

        all_instances_dir = os.listdir(self.data_root)
        for i, instance_dir in enumerate(all_instances_dir):
            for fname in os.listdir(os.path.join(self.data_root, instance_dir)):
                self.labels.append(i)
                img = Image.open(os.path.join(self.data_root, instance_dir, fname))
                self.images.append(img)
        zipped = list(zip(self.labels, self.images))
        random.shuffle(zipped)
        self.labels, self.images = zip(*zipped)
        self.n_classes = len(all_instances_dir)
        self.labels = torch.Tensor(self.labels)

    def __getitem__(self, idx):
        img = self.transforms(self.images[idx])
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.labels)
