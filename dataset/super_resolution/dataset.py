import os

import cv2
import numpy as np
import torch
import warnings
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from typing import Dict

from dataset.super_resolution import imgproc


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        gt_image_dir (str): Train/Valid dataset address.
        gt_image_size (int): Ground-truth resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(
            self,
            gt_image_dir: str,
            gt_image_size: int,
            upscale_factor: int,
            mode: str,
    ) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_file_names = [os.path.join(gt_image_dir, image_file_name) for image_file_name in
                                 os.listdir(gt_image_dir)]
        self.gt_image_size = gt_image_size
        self.upscale_factor = upscale_factor
        self.mode = mode

    def __getitem__(self, batch_index: int) -> [Dict[str, Tensor], Dict[str, Tensor]]:
        # Read a batch of image data
        gt_image = cv2.imread(self.image_file_names[batch_index]).astype(np.float32) / 255.

        h, w, _ = gt_image.shape
        if h < self.gt_image_size or w < self.gt_image_size:
            # print(gt_image.shape, self.image_file_names[batch_index])
            # warnings.warn("small size of gt_image")
            ngt_image = np.zeros((max(h, self.gt_image_size), max(w, self.gt_image_size), 3), dtype=np.float32)
            ngt_image[:h, :w, :] = gt_image
            gt_image = ngt_image

        # Image processing operations
        if self.mode == "Train":
            gt_crop_image = imgproc.random_crop(gt_image, self.gt_image_size)
        elif self.mode == "Valid":
            gt_crop_image = imgproc.center_crop(gt_image, self.gt_image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_crop_image = imgproc.image_resize(gt_crop_image, 1 / self.upscale_factor)

        # BGR convert RGB
        gt_crop_image = cv2.cvtColor(gt_crop_image, cv2.COLOR_BGR2RGB)
        lr_crop_image = cv2.cvtColor(lr_crop_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_crop_tensor = imgproc.image_to_tensor(gt_crop_image, False, False)
        lr_crop_tensor = imgproc.image_to_tensor(lr_crop_image, False, False)

        return {"gt": gt_crop_tensor, "lr": lr_crop_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TestImageDataset(Dataset):
    """Define Test dataset loading methods.
    Args:
        test_gt_images_dir (str): ground truth image in test image
        test_lr_images_dir (str): low-resolution image in test image
    """

    def __init__(self, test_gt_images_dir: str, test_lr_images_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.gt_image_file_names = [os.path.join(test_gt_images_dir, x) for x in os.listdir(test_gt_images_dir)]
        self.lr_image_file_names = [os.path.join(test_lr_images_dir, x) for x in os.listdir(test_lr_images_dir)]

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        gt_image = cv2.imread(self.gt_image_file_names[batch_index]).astype(np.float32) / 255.
        lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.

        # BGR convert RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor, "lr": lr_tensor}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)



def get_loader(args, device) -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(args.train_gt_images_dir,
                                            args.gt_image_size,
                                            args.upscale_factor,
                                            "Train")
    test_datasets = TestImageDataset(args.test_gt_images_dir, args.test_lr_images_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return train_prefetcher, test_prefetcher
