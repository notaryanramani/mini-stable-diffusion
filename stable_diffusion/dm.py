import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from .unet import UNet
from .vae import VAE_Encoder, VAE_Decoder
from .bert import BERT
from .clip import CLIP
from .utils import Config, rescale, TimePositionEmbedding
from .scheduler import Scheduler
import numpy as np
from typing import Union
from tqdm import tqdm


config = Config()


class DiffusionModel(nn.Module):
    def __init__(self, n_embd:int=config.TIME_POS_DIM) -> None:
        super().__init__()
        self.time_pos_emb = TimePositionEmbedding(n_embd=320, time_steps=1000)
        self.time_emb = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.SiLU(),
            nn.Linear(n_embd * 4, n_embd * 4),
            nn.SiLU()
        )
        self.bert = BERT()
        # self.clip = CLIP()
        self.unet = UNet()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.scheduler = Scheduler()

    def forward(self, x:torch.Tensor, noise:torch.Tensor, prompt:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        '''
            Shapes for Input
                x: (B, C, H, W)
                noise: (B, 4, LATENT_H, LATENT_H)
                prompt: (B, T)  where T <= 64
                time = (B) 
        '''
        # (B) -> (B, 320)
        time = self.time_pos_emb(time)
        # (B, 320) -> (B, 1280)
        time = self.time_emb(time)

        # (B, 3, 256, 256) -> (B, 4, 32, 32)
        x = self.encoder(x, noise)
        # (B, 4, 32, 32) -> (B, 1280, 1, 1) -> (B, 4, 32, 32)
        x = self.unet(x, prompt, time)
        # (B, 4, 32, 32) -> (B, 3, 256, 256)
        out = self.decoder(x)

        return out
    
    def train_model(self, 
              data_loader,
              epochs:int=200, 
              lr:float= 3e-4, 
              batch_size:int=32, 
              eta_min = 3e-6,
              warmup_epcohs = 50, # try to keep this 25% of epochs
              return_state_dict=False,
              autocast = False) -> Union[dict, None]:

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        opt_sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
        mse = nn.MSELoss()

        for epoch in range(epochs):
            if epoch > warmup_epcohs:
                opt_sch.step()
            pb = tqdm(range(len(data_loader)))
            pb.set_description(f'Train Epoch [{epoch+1}/{epochs}]: ')
            for _ in enumerate(pb):
                images, prompts = data_loader.get_batch()
                if torch.rand(1).item() < 0.1:
                    prompts = [''] * batch_size  # 10% chance of training without labels
                images = images.to(config.DEVICE)
                prompts_emb = self.bert(prompts).to(device=config.DEVICE)
                # clip prompts
                # prompts_emb = self.clip(prompts).to(device=config.DEVICE)
                t = self.scheduler.sample_timesteps(batch_size).to(config.DEVICE)
                x_t, noise = self.scheduler.noise_images(images, t)
                latent_noise = torch.randn(batch_size, 4, config.LATENT_H, config.LATENT_H).to(device=config.DEVICE)
                if autocast:
                    with torch.autocast(dtype=torch.bfloat16, device_type=config.DEVICE):
                        predicted_noise = self(x_t, latent_noise, prompts_emb, t)
                        loss = mse(noise, predicted_noise)
                else:
                    predicted_noise = self(x_t, latent_noise, prompts_emb, t)
                    loss = mse(noise, predicted_noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                opt_sch.step()
                pb.set_postfix({'Loss':loss.item()})
        if return_state_dict:
            return self.state_dict()


    def generate(self,
                 prompt:str,
                 u_prompt:Union[str, None] = None,
                 input_img:Union[Image.Image, None]=None,
                 strength:float=0.8,
                 cfg:bool=True,
                 cfg_weight:float=7.5,
                 n_inference_steps:int=50,
                 seed = None,
                ):

        with torch.no_grad():

            assert 0 < strength <= 1, 'Strength should be in between 0 and 1'

            generator = torch.Generator(device=self.device)
            if seed:
                generator.manual_seed(seed)
            else:
                generator.seed()

            cprompt_logits = self.bert(prompt).to(device=config.DEVICE)
            # cprompt_logits = self.clip(prompt).to(device=config.DEVICE)
            if cfg:
                uprompt_logits = self.bert(u_prompt).to(device=config.DEVICE)
                prompt_logits = torch.cat([cprompt_logits, uprompt_logits])
                # uprompt_logits = self.clip(u_prompt).to(device=config.DEVICE)
                # prompt_logits = torch.cat([cprompt_logits, uprompt_logits])

            else:
                prompt_logits = cprompt_logits

            latent_shape = (1, 4, config.LATENT_H, config.LATENT_H)

            if input_img:
                input_img_tensor = input_img.resize((config.LATENT_H, config.LATENT_W))
                input_img_tensor = np.array(input_img_tensor)
                input_img_tensor = torch.tensor(input_img_tensor, dtype = torch.int32, device=config.DEVICE)

                input_img_tensor = rescale(input_img_tensor, (0, 255), (-1, 1))

                # Batch Size = 1 if image is provided
                input_img_tensor = input_img_tensor.unsqueeze(0)

                # (1, H, W, C) -> (1, C, H, W)
                input_img_tensor = input_img_tensor.permute(0, 3, 1, 2)

                encoder_noise = torch.randn(latent_shape, generator=generator, device=self.device)
                latents = self.encoder(input_img, encoder_noise)

                self.scheduler.set_strength(strength=strength)
                latents, _ = self.scheduler.noise_images(latents, self.scheduler.timesteps[0])

            else:
                latents = torch.randn(latent_shape, generator=generator, device=self.device)

            self.scheduler.set_inference_timesteps(n_inference_steps)
            timesteps = tqdm(self.scheduler.timesteps)

            for step in timesteps:
                time_embedding = self.time_pos_emb(step).to(device=config.DEVICE)

                x = latents

                if cfg:
                    x = x.repeat(2, 1, 1, 1)
                y = self.unet(x, prompt_logits, time_embedding)

                if cfg:
                    y_cond, y_uncond = y.chunk(2)
                    y = (y_cond - y_uncond) * cfg_weight + y_uncond

                latents = self.scheduler.step(step, latents, y)

            images = self.decoder(latents)

            images = rescale(images, (-1, 1), (0, 255))

            # (B, C, H, W) -> (B, H, W, C)
            images = images.permute(0, 2, 3, 1)
            images = images.detach().numpy()

            return images[0]
