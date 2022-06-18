import torch
from torch.utils.data import DataLoader
from util import compute_psnr_and_ssim
from model.Unet import UNet
from dataset.RESIDE_ITS import BasicDataset
import time
from tqdm.auto import tqdm
from kornia.color.hsv import rgb_to_hsv
from kornia.color.ycbcr import rgb_to_ycbcr
from kornia.color.lab import rgb_to_lab
from PIL import Image

IMGDIR = "result/RESIDE SOTS Outdoor"

if IMGDIR == "result/HAZE RD":
    path = "dataset/HazeRD/data"
elif IMGDIR == "result/D Hazy":
    path = "dataset/D_Hazy"
elif IMGDIR == "result/RESIDE SOTS Outdoor":
    path = "dataset/RESIDE_SOTS/outdoor"
elif IMGDIR == "result/RESIDE SOTS Indoor":
    path = "dataset/RESIDE_SOTS/indoor"
elif IMGDIR == "result/HSTS":
    path = "dataset/HSTS"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save = False
i = 0

if __name__ == "__main__":

    total_psnr = 0
    total_ssim = 0

    Hazy = path + "/hazy"
    Clear = path + "/clear"

    checkpoint = torch.load(f"./cpt/Final Outdoor Dehazing Weights.pth")
    net = UNet()
    net = net.to(DEVICE)
    net.load_state_dict(checkpoint)
    net.eval()
    dataset = BasicDataset(Hazy, Clear)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # torch.cuda.synchronize()
    # start = time.time()

    for patch, mask in tqdm(test_loader):
        patch = patch.to(DEVICE)
        patch = torch.cat((patch, rgb_to_hsv(patch), rgb_to_ycbcr(patch), rgb_to_lab(patch)), dim=1)
        ou = net.forward(patch)
        dehaze = ou.squeeze(0).mul_(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        gt = mask.squeeze(0).mul_(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        psnr, ssim = compute_psnr_and_ssim(dehaze, gt)

        total_psnr += psnr
        total_ssim += ssim

        if save:
            ou_save = Image.fromarray(dehaze)
            ou_save.save(IMGDIR + f"/dehaze_{dataset.hazes[i]}.png")

        i += 1

    # torch.cuda.synchronize()
    # end = time.time()
    # elasped = end - start
    #
    # print(elasped / 500)

    print("PSNR = ", psnr)
    print("SSIM = ", ssim)