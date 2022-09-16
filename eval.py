from utils import p_sample_loop
from unet import UNet
import torch
from torchvision.utils import save_image
import os

device = torch.device("cuda")


def generate():
    f_name = 'checkpoint-epoch1720-1d14bp9b'
    model = UNet()
    ckpt = torch.load('./checkpoints/' + f_name + '.pth')
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    x = torch.randn((64, 3, 32, 32), device=device)

    res = p_sample_loop(model, x, s=100, n=0, DDIM=True)

    res = [(x + 1) / 2 for x in res]

    os.makedirs('./images/DDIM' + f_name, exist_ok=True)
    for i, x in enumerate(res):
        save_image(x, './images/DDIM' + f_name + '/' + str(i + 1) + '.png')


if __name__ == '__main__':
    generate()
