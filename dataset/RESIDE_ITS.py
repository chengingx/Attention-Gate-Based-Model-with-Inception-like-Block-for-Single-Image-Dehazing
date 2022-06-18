from os.path import splitext
from os import listdir
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.hazes = listdir(imgs_dir)

    def __len__(self):
        return len(self.hazes)

    def preprocess(self, img, mode="none"):
        transform_size = transforms.Resize((256, 256))
        img = transform_size(img)

        if mode == "minmax":
            min = torch.min(img)
            max = torch.max(img)
            transformed_img = (img - min) / (max - min)
            return transformed_img

        if mode == "none":
            transformed_img = img / 255.0
            return transformed_img

        return False

    def __getitem__(self, i):
        gt_img_path = self.masks_dir + "/" + splitext(self.hazes[i])[0].split('_')[0] + ".png"
        hazy_img_path = self.imgs_dir + "/" + self.hazes[i]

        gt_img = torchvision.io.read_image(gt_img_path)
        hazy_img = torchvision.io.read_image(hazy_img_path)

        hazy = self.preprocess(hazy_img, "minmax")
        gt = self.preprocess(gt_img, "minmax")

        # When train indoor dataset, use the following code to do data agumentation
        # hflip = random.random() < 0.5
        # transform_hflip = transforms.RandomHorizontalFlip(p=1)
        # hazy = transform_hflip(hazy)
        # gt = transform_hflip(gt)

        return hazy, gt
