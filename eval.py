from utils import p_sample_loop
from unet import UNet
import torch
from torchvision import transforms
from torchvision.utils import save_image
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device("cuda")


def generate():
    model = UNet()
    model.to(device)

    f_info = 'epoch1720-1d14bp9b'
    checkpoint = torch.load('./checkpoints/checkpoint-' + f_info + '.pth')
    model.load_state_dict(checkpoint['model'])

    x = torch.randn((64, 3, 32, 32), device=device)

    res = p_sample_loop(model, x, False)

    # TODO
    # clip

    res = [transforms.Normalize((-1, -1, -1), (2, 2, 2))(x) for x in res]

    os.makedirs('./images/' + f_info, exist_ok=True)
    for i in range(len(res)):
        save_image(res[i], './images/' + f_info + '/' + str(i) + '.png')


if __name__ == '__main__':
    generate()
