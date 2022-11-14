import os

import h5py
import numpy as np
import paddle
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from paddle.vision import transforms

from dataloader.transforms import Rotation


class NyuDepth(paddle.io.Dataset):
    def __init__(self, root_dir, split, csv_file, n_sample=500):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.samples = pd.read_csv(os.path.join(root_dir, csv_file))
        self.n_sample = n_sample
        self._s = np.random.uniform(1.0, 1.5)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.samples.iloc[idx, 0])
        rgb_h5, depth_h5 = self.load_h5(file_name)
        rgb_image = Image.fromarray(rgb_h5, mode='RGB')
        depth_image = Image.fromarray(depth_h5.astype('float32'), mode='F')
        rgb_transform, depth_transform = self.get_transform()
        rgb_image = rgb_transform(rgb_image)
        depth_image = depth_transform(depth_image)
        depth_image = depth_image / self._s
        sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
        rgbd_image = paddle.concat((rgb_image, sparse_image), 0)
        sample = {'rgbd': rgbd_image, 'depth': depth_image}
        return sample

    def get_transform(self):
        s = int(240 * self._s)
        degree = np.random.uniform(-5.0, 5.0)

        if self.split == 'train':
            rgb_transform = transforms.Compose([
                transforms.Resize(size=s),
                Rotation(degree),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(size=(228, 304)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            depth_transform = transforms.Compose([
                transforms.Resize(size=s),
                Rotation(degree),
                transforms.CenterCrop(size=(228, 304)),
                transforms.ToTensor(),
            ])
            return rgb_transform, depth_transform
        else:
            rgb_transform = transforms.Compose([
                transforms.Resize(size=(228, 304)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            depth_transform = transforms.Compose([
                transforms.Resize(size=(228, 304)),
                transforms.ToTensor(),
            ])
            return rgb_transform, depth_transform

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
        rgb = f['rgb'][:].transpose(1, 2, 0)
        depth = f['depth'][:]
        return rgb, depth

    def createSparseDepthImage(self, depth_image, n_sample):
        random_mask = paddle.zeros((1, depth_image.shape[1], depth_image.shape[2]))
        n_pixels = depth_image.shape[1] * depth_image.shape[2]
        # n_valid_pixels = paddle.sum(depth_image > 0.0001)
        # #        print('===> number of total pixels is: %d\n' % n_pixels)
        # #        print('===> number of total valid pixels is: %d\n' % n_valid_pixels)
        perc_sample = n_sample / n_pixels
        random_mask = paddle.bernoulli(paddle.ones_like(random_mask) * perc_sample)
        sparse_depth = depth_image * random_mask
        return sparse_depth


if __name__ == '__main__':
    dataset = NyuDepth(root_dir='../data/nyudepth_hdf5', split='train',
                       csv_file='train.csv')
    sample = dataset[0]
    print(len(dataset))
    print(sample['rgbd'].shape)
    print(sample['depth'].shape)
    color_raw = sample['rgbd'][0:3, :, :]
    depth_raw = sample['rgbd'][3, :, :]
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(color_raw.numpy().transpose(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(depth_raw.numpy())
    plt.show()
