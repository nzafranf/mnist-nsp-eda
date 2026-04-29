from flow_matching.path import CondOTProbPath
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embedding_l = nn.Linear(3, dim)

    def forward(self, time):
        device = time.device
        time = torch.cat([time, torch.cos(time), torch.sin(time)], dim=-1)
        return self.embedding_l(time)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, c=64):
        super().__init__()
        
        # Time embeddings for each skip connection
        self.time_mlp1 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, c)
        )
        
        self.time_mlp2 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 2*c),
            nn.SiLU(),
            nn.Linear(2*c, 2*c)
        )
        
        self.time_mlp3 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 4*c),
            nn.SiLU(),
            nn.Linear(4*c, 4*c)
        )
        
        # Encoder
        self.conv1 = DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(c, 2*c)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(2*c, 4*c)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(4*c, 8*c)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(8*c, 4*c, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(12*c, 4*c)  # 4c + 4c + 4c (skip + up + time)
        self.upconv2 = nn.ConvTranspose2d(4*c, 2*c, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(6*c, 2*c)   # 2c + 2c + 2c
        self.upconv1 = nn.ConvTranspose2d(2*c, c, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(3*c, c)     # c + c + c
        
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    
    def forward(self, x, t):
        # Time embeddings
        t = t.view(-1, 1)
        if t.numel() == 1:
            t = t.expand(x.shape[0], -1)
        t1 = self.time_mlp1(t)
        t2 = self.time_mlp2(t)
        t3 = self.time_mlp3(t)

        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)

        # Decoder with proper dimension handling
        # Step 1
        up3 = self.upconv3(conv4)

        # Ensure x is 4D (it should be [batch, channels, height, width])
        if x.dim() != 4:
            raise ValueError(f"Expected x to be 4D, got shape {x.shape}")

        # Pad if needed to match spatial dimensions
        if up3.shape[-2:] != conv3.shape[-2:]:
            pad_h = max(0, conv3.size(-2) - up3.size(-2))
            pad_w = max(0, conv3.size(-1) - up3.size(-1))
            up3 = F.pad(up3, (0, pad_w, 0, pad_h), mode='replicate')

        # Reshape time embedding
        t_emb3 = t3.view(t3.size(0), t3.size(1), 1, 1)
        t_emb3 = t_emb3.expand(-1, -1, up3.size(2), up3.size(3))
        concat3 = torch.cat([up3, conv3, t_emb3], dim=1)
        conv5 = self.conv5(concat3)

        # Step 2
        up2 = self.upconv2(conv5)
        if up2.shape[-2:] != conv2.shape[-2:]:
            pad_h = max(0, conv2.size(-2) - up2.size(-2))
            pad_w = max(0, conv2.size(-1) - up2.size(-1))
            up2 = F.pad(up2, (0, pad_w, 0, pad_h), mode='replicate')
        t_emb2 = t2.view(t2.size(0), t2.size(1), 1, 1)
        t_emb2 = t_emb2.expand(-1, -1, up2.size(2), up2.size(3))
        concat2 = torch.cat([up2, conv2, t_emb2], dim=1)
        conv6 = self.conv6(concat2)

        # Step 3
        up1 = self.upconv1(conv6)
        if up1.shape[-2:] != conv1.shape[-2:]:
            pad_h = max(0, conv1.size(-2) - up1.size(-2))
            pad_w = max(0, conv1.size(-1) - up1.size(-1))
            up1 = F.pad(up1, (0, pad_w, 0, pad_h), mode='replicate')
        t_emb1 = t1.view(t1.size(0), t1.size(1), 1, 1)
        t_emb1 = t_emb1.expand(-1, -1, up1.size(2), up1.size(3))
        concat1 = torch.cat([up1, conv1, t_emb1], dim=1)
        conv7 = self.conv7(concat1)

        return self.final_conv(conv7)