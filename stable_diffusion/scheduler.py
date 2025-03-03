import torch
from .utils import Config
from typing import Tuple


config = Config()


class Scheduler:
    def __init__(self, 
                 noise_steps:int = 1000,
                 beta_start:float = 0.00085,
                 beta_end:float = 0.012) -> None:
        self.noise_steps = noise_steps

        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device=config.DEVICE) ** 2
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.timesteps = torch.flip(torch.arange(0, noise_steps), [0]).clone().to(device=config.DEVICE)
        self.n_inference_steps = 1

    def noise_images(self, x:torch.Tensor, t:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ah_sqrt = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        ah_om_sqrt = torch.sqrt(1. - self.alpha_hat[t]).view(-1, 1, 1, 1)
        epsilon = torch.rand_like(x)
        x = ah_sqrt * x + ah_om_sqrt * epsilon
        return x, epsilon
    
    def sample_timesteps(self, n:int) -> torch.Tensor:
        return torch.randint(1, self.noise_steps, (n,))
    
    def set_inference_timesteps(self, n_inference_steps:int) -> None:
        self.n_inference_steps = n_inference_steps
        step_ratio = self.timesteps // n_inference_steps
        self.timesteps = torch.arange(0, n_inference_steps)[::-1].clone().to(dtype=torch.int32, device=config.DEVICE) * step_ratio

    def set_strength(self, strength:float = 1.) -> None:
        start = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = self.timesteps[start:]

    def step(self, x:torch.Tensor, t:torch.Tensor, y:torch.Tensor) -> torch.Tensor:

        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        noise = torch.randn_like(x) if t>1 else torch.zeros_like(x)

        x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1-alpha_hat))) * y) + torch.sqrt(beta) * noise
        return x


