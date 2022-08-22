import os
import sys
import wandb
import yaml

import torchvision
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torchvision import transforms
from dataset import MyDataset
from unet import UNet
from loss import Loss


def init_wandb(config):
    id = wandb.util.generate_id()
    if config["resume"]:
        id = config["id"]
    wandb.init(project="diffusion_model", config=config, id=id, resume="allow")
    return id


def create_dir(config):
    if not os.path.exists(config["checkpoint_path"]):
        os.makedirs(config["checkpoint_path"])
    if not os.path.exists(config["model_save_path"]):
        os.makedirs(config["model_save_path"])
    if not os.path.exists(config["image_save_path"]):
        os.makedirs(config["image_save_path"])
    if not os.path.exists('./logs'):
        os.makedirs('./logs')


def main():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    create_dir(config)

    id = 0
    if config['wandb']:
        id = init_wandb(config)
    log = open(f'./logs/log-{id}.txt', 'a')
    sys.stdout = log

    train(config, id)


def train(config, id):
    resume = config["resume"]
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    epochs = config['epochs']
    lr = config['learning_rate']
    image_size = config['image_size']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_steps = config['time_steps']

    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # get dataset
    # train_data = MyDataset(root='C:/Users/wang/data/anime_face', transform=transform)
    train_data = torchvision.datasets.CIFAR10(config['data_path'], download=True, train=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # get model
    model = UNet()
    model.to(device)
    # get optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    # get loss function
    criterion = Loss()

    start = 0
    if resume:
        checkpoint = torch.load(config['checkpoint_path'] + '')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start}")

    for epoch in range(start, epochs):
        for i, (img, _) in enumerate(train_loader):
            img = img.to(device)

            t = torch.randint(time_steps, (img.shape[0],), device=device)

            loss = criterion(model, img, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % config['print_steps'] == 0:
                print(f'Epoch [{epoch + 1}/ {epochs}], loss: {loss.item()}')
                sys.stdout.flush()
                wandb.log({'loss': loss.item()}, step=epoch * len(train_loader) + i + 1)

        if (epoch + 1) % config['checkpoint_save_epochs'] == 0:
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(save_dict, config['checkpoint_path'] + f'/checkpoint-epoch{epoch + 1}-{id}.pth')
            print('saved checkpoint!')

        # if (epoch + 1) % config['image_save_epochs'] == 0:
        #     out = model(img)
        #     save_image(out.reshape(-1, 1, 28, 28), config['image_save_path'] + f'/image-epoch{epoch + 1}-{id}.png')
        #     wandb.log({'img': wandb.Image(config['image_save_path'] + f'/image-epoch{epoch + 1}-{id}.png')})

    torch.save(model.state_dict(), wandb.run.dir + '/' + f'/model-latest-{id}.pth')
    wandb.save(wandb.run.dir + '/' + f'/model-latest-{id}.pth')
    wandb.save('config.yaml')
    print('saved model!')


if __name__ == '__main__':
    main()
