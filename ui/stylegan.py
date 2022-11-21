import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import resolution_to_index
import math
import random

class PixelNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
  
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2) + self.epsilon)


class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias_value=0, lr_mul=1):
        super().__init__()
        self.lr_mul = lr_mul

        self.weights = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias_value:
            self.bias = nn.Parameter(torch.ones(out_dim))
        else:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self.scale = 1 / math.sqrt(in_dim)
    
    def forward(self, x):
        return F.linear(x, self.weights * self.scale * self.lr_mul, self.bias * self.lr_mul)


class EqualizedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding

        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        fan_in = in_channels * (kernel ** 2)
        self.scale = 1 / math.sqrt(fan_in)

    def forward(self, x):
        return F.conv2d(x, self.weights * self.scale, self.bias, stride=self.stride, padding=self.padding)


class Mapping(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()

        blocks = [PixelNormalization()]
        blocks += [EqualizedLinear(z_dim, w_dim, lr_mul=0.01), nn.LeakyReLU(0.2)]

        for i in range(7):
            blocks += [EqualizedLinear(w_dim, w_dim, lr_mul=0.01), nn.LeakyReLU(0.2)]

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv(in_channels, in_channels, kernel=3)
        self.conv2 = EqualizedConv(in_channels, out_channels, kernel=3)

        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        return x


class AdaIN(nn.Module):
    def __init__(self, w_dim, channels):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.scale = EqualizedLinear(w_dim, channels)
        self.bias = EqualizedLinear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)
        y_scale = self.scale(w)[:, :, None, None]
        y_bias = self.bias(w)[:, :, None, None]
        
        return y_scale * x + y_bias


class GeneratorBlock(nn.Module):
    def __init__(self, w_dim, in_channels, out_channels, device):
        super().__init__()
        self.device = device
        self.conv1 = EqualizedConv(in_channels, out_channels, kernel=3)
        self.conv2 = EqualizedConv(out_channels, out_channels, kernel=3)
        self.adain1 = AdaIN(w_dim, out_channels)
        self.adain2 = AdaIN(w_dim, out_channels)
        self.noise_scale1 = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        n_batch, _, height, width = x.shape

        x = self.conv1(x)
        x = self.noise_scale1 * torch.randn(n_batch, 1, height, width, device=self.device) + x
        x = self.activation(x)
        x = self.adain1(x, w)
        x = self.conv2(x)
        x = self.noise_scale2 * torch.randn(n_batch, 1, height, width, device=self.device) + x
        x = self.activation(x)
        x = self.adain2(x, w)

        return x


class Generator(nn.Module):
    def __init__(self, resolution, z_dim, w_dim, device, config, start_resolution=4):
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.const_input = nn.Parameter(torch.ones(1, config["channels_per_layer"][0], start_resolution, start_resolution))
        self.mapping = Mapping(z_dim, w_dim)

        self.blocks = nn.ModuleList([])
        self.toRGB_blocks = nn.ModuleList([EqualizedConv(config["channels_per_layer"][1], 3, kernel=1, padding=0)])

        self.n_blocks = resolution_to_index(resolution) - 1

        in_channels = config["channels_per_layer"][0]
        out_channels = config["channels_per_layer"][1]
        self.conv1 = EqualizedConv(in_channels, out_channels, kernel=3)
        self.adain1 = AdaIN(w_dim, out_channels)
        self.adain2 = AdaIN(w_dim, out_channels)
        self.noise_scale1 = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        self.activation = nn.LeakyReLU(0.2)
        self.upsample = Upsample()

        for i in range(1, self.n_blocks+1):
            in_channels = config["channels_per_layer"][i]
            out_channels = config["channels_per_layer"][i+1]

            self.blocks.append(GeneratorBlock(w_dim, in_channels, out_channels, device))
            self.toRGB_blocks.append(EqualizedConv(out_channels, 3, kernel=1, padding=0))
    
    def first_block(self, x, w):
        n_batch, _, height, width = x.shape

        x = self.noise_scale1 * torch.randn(n_batch, 1, height, width, device=self.device) + x
        x = self.adain1(x, w)
        x = self.conv1(x)
        x = self.noise_scale2 * torch.randn(n_batch, 1, height, width, device=self.device) + x
        x = self.activation(x)
        x = self.adain2(x, w)
        return x
    
    def get_w(self, z):
        return self.mapping(z)
    
    def w_mean(self, n_samples=1000):
        noise = torch.randn(n_samples, self.z_dim, device=self.device)
        l_mean = self.mapping(noise).mean(0, keepdim=True)

        return l_mean

    def forward(self, styles, alpha, current_resolution, fade_in, index=None, return_w=False, truncation=1, truncation_w=None, is_input_w=False):
        n_stages = resolution_to_index(current_resolution) - 1

        if not is_input_w:
            if isinstance(styles, list) and len(styles) == 2:
                styles = [self.mapping(style) for style in styles]
                if not index:
                    index = random.randint(1, n_stages)
                style1 = styles[0].unsqueeze(0).repeat(index, 1, 1)
                style2 = styles[1].unsqueeze(0).repeat(n_stages + 1 - index, 1, 1)

                styles = torch.cat([style1, style2], 0)
            else:
                styles = self.mapping(styles)
                styles = styles.unsqueeze(0).repeat(n_stages + 1, 1, 1)
        else:
            styles = styles.unsqueeze(0).repeat(n_stages + 1, 1, 1)
        
            
        if truncation != 1:
            temp_styles = []

            for style in styles:
                temp_styles.append(truncation_w + truncation*(style - truncation_w))

            styles = torch.stack(temp_styles)
        
        out = self.first_block(self.const_input, styles[0, :, :])

        if current_resolution == 4:
            out = self.toRGB_blocks[0](out)
            if return_w: return self.activation(out), styles
            else: return self.activation(out)

        for i in range(n_stages):
            out = self.upsample(out)
            skip = out
            if i == n_stages-1:
                out = self.blocks[i](out, styles[i+1, :, :])
            else:
                out = self.blocks[i](out, styles[i+1, :, :])


        skip = self.toRGB_blocks[n_stages-1](skip)
        out = self.toRGB_blocks[n_stages](out)

        if fade_in:
            out = (1 - alpha) * skip + alpha * out

        if return_w: return out, styles
        else: return out


class Discriminator(nn.Module):
    def __init__(self, resolution, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.fromRGB_blocks = nn.ModuleList([EqualizedConv(3, config["channels_per_layer"][0], kernel=1, padding=0)])

        self.n_blocks = resolution_to_index(resolution) - 1

        self.activation = nn.LeakyReLU(0.2)
        self.downsample = Downsample()

        self.last_block = nn.Sequential(
            EqualizedConv(config["channels_per_layer"][1]+1, config["channels_per_layer"][1], kernel=3),
            nn.LeakyReLU(0.2),
            EqualizedConv(config["channels_per_layer"][1], config["channels_per_layer"][0], kernel=4, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.linear = EqualizedLinear(config["channels_per_layer"][0], 1)

        for i in range(1, self.n_blocks+1):
            in_channels = config["channels_per_layer"][i+1]
            out_channels = config["channels_per_layer"][i]

            self.blocks.append(ConvBlock(in_channels, out_channels))
            self.fromRGB_blocks.append(EqualizedConv(3, in_channels, kernel=1, padding=0))
    
    def batch_std(self, x):
        batch_std = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
       
        return torch.cat([x, batch_std], dim=1)

    def forward(self, x, alpha, current_resolution, fade_in):
        if current_resolution == 4:
            out = self.fromRGB_blocks[0](x)
            out = self.activation(out)
            out = self.batch_std(out)
            out = self.last_block(out)
            out = self.linear(out.view(out.shape[0], -1))

            return out

        n_stages = resolution_to_index(current_resolution) - 1

        skip = self.downsample(x)
        skip = self.fromRGB_blocks[n_stages-1](skip)
        skip = self.activation(skip)

        out = self.fromRGB_blocks[n_stages](x)
        out = self.activation(out)
        out = self.blocks[n_stages-1](out)
        out = self.downsample(out)

        if fade_in:
            out = (1 - alpha) * skip + alpha * out

        for i in reversed(range(n_stages-1)):
            out = self.blocks[i](out)
            out = self.downsample(out)

        out = self.batch_std(out)
        out = self.last_block(out)
        out = self.linear(out.view(out.shape[0], -1))
        return out


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        self.pad = nn.ReplicationPad2d(1)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(-1, 1, height, width)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(batch_size, channels, height, width)


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.blur = Blur()
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.blur(x)
        
        return x


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.blur = Blur()
    
    def forward(self, x):
        x = self.blur(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        
        return x
