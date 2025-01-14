import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, in_channels, H, W)

        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)  # (never used it !!!!)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, channels, H, W)

        residue = x

        n,c,h,w = x.shape

        # (Batch, channels, H, W) -> (Batch, channels, H*W)
        x = x.view(n,c,h*w)

        # (Batch, channels, H*W) -> (Batch, H*W, channels)
        x = x.transpose(-1,-2)

        # (Batch, H*W, channels) -> (Batch, H*W, channels)
        x = self.attention(x)

        # (Batch, H*W, channels) -> (Batch, channels, H*W)
        x = x.transpose(-1,-2)

        # (Batch, channels, H*W) -> (Batch, channels, H, W)
        x = x.view(n,c,h,w) 

        x += residue

        return x

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),
            
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch, 512, H/4, W/4) -> (Batch, 512, H/2, W/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch, 256, H/2, W/2) -> (Batch, 256, H, W)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            
            # (Batch, 128, H, W) -> (Batch, 3, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 4, H/8, W/8)

        x /= 0.18215

        for module in self:
            x = module(x)
        
        return x


        