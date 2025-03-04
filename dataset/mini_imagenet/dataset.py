import os
import json
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset


class MiniImageNet(Dataset):
    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 transform=None):
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i) for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)

        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    train_dataset = MiniImageNet(root_dir=r'../../train/train_mini_imagenet/mini_imagenet',
                                 csv_name="new_train.csv",
                                 json_path=r'../../train/train_mini_imagenet/mini_imagenet/classes_name.json',
                                 transform=transforms.Compose(
                                  [
                                      transforms.ToTensor()
                                  ]
                              )
                                 )

    classes_name = json.load(open(r'../../train/train_mini_imagenet/mini_imagenet/classes_name.json', 'r'))
    id_2_cls = dict((int(v[0]), v[1]) for k, v in classes_name.items())

    def show(index):
        img, label = train_dataset[index]
        plt.imshow(img.permute(1, 2, 0))
        plt.title(id_2_cls[label])
        plt.show()
        plt.close()

    show(2)
