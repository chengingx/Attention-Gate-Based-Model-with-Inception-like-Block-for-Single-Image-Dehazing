import torch.nn as nn
from kornia.color.hsv import rgb_to_hsv
from kornia.color.ycbcr import rgb_to_ycbcr
from kornia.color.lab import rgb_to_lab
import config
from model.Unet import UNet
from dataset.OTS_BETA import BasicDataset
import torch
import numpy as np
import pytorch_ssim

Hazy = config.Hazy
Clear = config.Clear

dataset = BasicDataset(Hazy, Clear)

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(m.bias, std=0.001)


class MS_SSIM_L2_LOSS(nn.Module):
    def __init__(self):
        super(MS_SSIM_L2_LOSS, self).__init__()
        self.L2 = torch.nn.MSELoss()
        self.ssim = pytorch_ssim.SSIM()

    def __call__(self, x, y):
        SSIM = 0.8 * (1 - self.ssim(x, y)) + 0.2 * self.L2(x,y)
        return SSIM


if __name__ == '__main__':

    train_indices = None
    test_indices = None

    print("Record dataset indices...")
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.seed(2021)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    f = open('train_indices.txt', 'w')
    f.write(str(train_indices))
    f.close()
    f = open('test_indices.txt', 'w')
    f.write(str(test_indices))
    f.close()

    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=2, pin_memory=True, sampler=train_sampler)
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True, sampler=test_sampler)
    print("Number of training/test patches:", (len(train_indices), len(test_indices)))

    import gc

    net = None
    gc.collect()

    net = UNet()
    net = net.to(config.DEVICE)

    if config.DEVICE == "cuda":
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True

    if config.PRETRAINED:
        print(f"Load cpt/Test Indoor {config.PRETRAINED_EPOCH}.pth")
        checkpoint = torch.load(f"cpt/Test Indoor {config.PRETRAINED_EPOCH}.pth")
        net.load_state_dict(checkpoint)

    from tqdm import trange

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss = MS_SSIM_L2_LOSS().to(config.DEVICE)
    epochs = config.NUM_EPOCHS - config.INITIAL_EPOCH
    bar = trange(epochs, desc="ML")

    for epoch in bar:
        epoch = epoch + 1 + config.INITIAL_EPOCH
        total_train_loss = 0
        total_test_loss = 0
        net.train()
        num_t = 0
        for patch, mask in train_loader:
            patch = patch.to(config.DEVICE)
            patch = torch.cat((patch, rgb_to_hsv(patch), rgb_to_ycbcr(patch), rgb_to_lab(patch)), dim=1)
            mask = mask.to(config.DEVICE)
            ou = net.forward(patch)
            train_loss = loss(mask.squeeze(1), ou)
            bar.set_description("Train (Loss = %g)" % train_loss)
            total_train_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            num_t += 1

        num_v = 0
        for patch, mask in test_loader:
            with torch.no_grad():
                patch = patch.to(config.DEVICE)
                patch = torch.cat((patch, rgb_to_hsv(patch), rgb_to_ycbcr(patch), rgb_to_lab(patch)), dim=1)
                mask = mask.to(config.DEVICE)
                ou = net.forward(patch)
                test_loss = loss(mask.squeeze(1), ou)
                bar.set_description("Test (Loss = %g)" % test_loss)
                total_test_loss += test_loss.item()
                num_v += 1

        average_train_loss = total_train_loss / num_t
        average_test_loss = total_test_loss / num_v

        Loss0 = np.array(average_train_loss)
        Loss1 = np.array(average_test_loss)
        np.save("./cpt/epoch_train{}".format(epoch), Loss0)
        np.save("./cpt/epoch_test{}".format(epoch), Loss1)

        torch.save(net.state_dict(), "./cpt/Outdoor {}.pth".format(epoch))