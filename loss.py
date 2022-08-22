import torch
from utils import q_sample


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, model, x_0, t):
        noise = torch.randn(x_0.shape, device=t.device)
        x_diffusion = q_sample(x_0, t, noise)
        predict_noise = model(x_diffusion, t)
        
        # print(predict_noise, noise)

        return torch.mean((noise - predict_noise) ** 2)
