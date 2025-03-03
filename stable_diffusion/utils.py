from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

@dataclass
class Config:
    VAE_INIT_CHANNELS:int = 128     # initial channels size for VAE
    UNET_INIT_CHANNELS:int = 320    # intiial channels size for UNet
    TIME_DIM:int = 1280             # time emebddings dimension
    TIME_POS_DIM:int = 320          # postional time embedding dimension
    PROMPT_DIM:int = 768            # prompt embeddings dimension (i.e. BERT embedding dimensions)
    IMG_SIZE:tuple = (256, 256)
    BATCH_SIZE:int = 2
    DROPOUT:float = 0.2
    LATENT_H:int = 32               # latent height
    LATENT_W:int = 32               # latent weight
    DEVICE:str = 'cuda' if torch.cuda.is_available() else 'cpu' 


class TimePositionEmbedding(nn.Module):

    ''' Position Embeddings for Time '''

    def __init__(self, n_embd:int, time_steps:int) -> None:
        super().__init__()
        self.pe = torch.zeros(time_steps, n_embd)
        pos = torch.arange(0, time_steps).float().unsqueeze(1)
        div_term = torch.tensor([10000.0]) ** (torch.arange(0, n_embd, 2) / n_embd)
        self.pe[:, 0::2] = torch.sin(pos / div_term)
        self.pe[:, 1::2] = torch.cos(pos / div_term)
        

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        out = self.pe.to(t.device)[t].view(-1, 320)
        return out
    

# function to convert [0, 255] to [-1, 1] and vice versa
def rescale(x, old, new, clamp=False, to_image=False):
    old_min, old_max = old
    new_min, new_max = new
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    if to_image:
        x = x.to(torch.uint8)
    return x
