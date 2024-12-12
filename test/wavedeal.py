import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch import nn

def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH
# 输入通道等于输出通道
class WavePool2(nn.Module):
    def __init__(self, in_channels):
        super(WavePool2, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav_two(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        elif self.option_unpool =='sumall':
            return LL + LH + HL + HH
        else:
            raise NotImplementedError

def wavelet_decomposition(image_path, output_path='output_wavelet_decomp.png'):
    # Load image and ensure it's in RGB mode
    original_image = Image.open(image_path).convert('RGB')
    
    # Convert the image to a tensor and add batch dimension
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Converts image to [0,1] range
    ])
    image_tensor = preprocess(original_image).unsqueeze(0)  # Add batch dimension
# 34_3_low_freq_real
    # Perform Wavelet decomposition using WavePool2
    pool = WavePool2(in_channels=3)
    LL, LH, HL, HH = pool(image_tensor)
    unpool=WaveUnpool(in_channels=3,option_unpool='sum')
    rec=unpool(LL, LH, HL, HH)

    # Prepare for plotting: convert tensors back to numpy arrays and squeeze batch dimension
    images = [
        original_image,
        transforms.ToPILImage()(LL.squeeze(0)).convert('RGB'),
        transforms.ToPILImage()(LH.squeeze(0)).convert('RGB'),
        transforms.ToPILImage()(HL.squeeze(0)).convert('RGB'),
        transforms.ToPILImage()(HH.squeeze(0)).convert('RGB'),
        transforms.ToPILImage()(rec.squeeze(0)).convert('RGB'),
        # transforms.ToPILImage()(ss.squeeze(0)).convert('RGB')
    ]
    
    titles = ['Original Image', 'Approximation', 'Horizontal detail',
              'Vertical detail', 'Diagonal detail', 'Reconstructed Image']

    # Plot original and decomposed images in a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axs.flat):
        if i < len(images):  # Only plot as many images as we have
            ax.imshow(images[i])
            # ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i])
            ax.set_axis_off()

    plt.tight_layout()
    
    # Save the figure to a file
    plt.savefig(output_path)
    
    # Show the plot
    plt.show()
# def wavelet_decomposition(image_path, output_path='output_wavelet_decomp.png'):
#     # Load image and ensure it's in RGB mode
#     LL = Image.open('real_B_LL.png').convert('RGB')
#     LH = Image.open('real_B_LH.png').convert('RGB')
#     HL = Image.open('real_B_HL.png').convert('RGB')
#     HH = Image.open('real_B_HH.png').convert('RGB')

#     # Convert the image to a tensor and add batch dimension
#     preprocess = transforms.Compose([
#         transforms.ToTensor(),  # Converts image to [0,1] range
#     ])
#     LL = preprocess(LL).unsqueeze(0)  # Add batch dimension
#     LH = preprocess(LH).unsqueeze(0)  # Add batch dimension
#     HL = preprocess(HL).unsqueeze(0)  # Add batch dimension
#     HH = preprocess(HH).unsqueeze(0)  # Add batch dimension
# # 34_3_low_freq_real
#     # Perform Wavelet decomposition using WavePool2
#     unpool=WaveUnpool(in_channels=3,option_unpool='sum')
#     rec=unpool(LL, LH, HL, HH)

#     # Prepare for plotting: convert tensors back to numpy arrays and squeeze batch dimension
#     images = [
#         # original_image,
#         transforms.ToPILImage()(LL.squeeze(0)).convert('RGB'),
#         transforms.ToPILImage()(LH.squeeze(0)).convert('RGB'),
#         transforms.ToPILImage()(HL.squeeze(0)).convert('RGB'),
#         transforms.ToPILImage()(HH.squeeze(0)).convert('RGB'),
#         transforms.ToPILImage()(rec.squeeze(0)).convert('RGB'),
#         # transforms.ToPILImage()(ss.squeeze(0)).convert('RGB')
#     ]
    
#     titles = ['Approximation', 'Horizontal detail',
#               'Vertical detail', 'Diagonal detail', 'Reconstructed Image']

#     # Plot original and decomposed images in a 2x3 grid
#     fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#     for i, ax in enumerate(axs.flat):
#         if i < len(images):  # Only plot as many images as we have
#             ax.imshow(images[i])
#             ax.set_title(titles[i])
#             ax.set_axis_off()

#     plt.tight_layout()
    
#     # Save the figure to a file
#     plt.savefig(output_path)
    
#     # Show the plot
#     plt.show()
# wavelet_decomposition('18_1.PNG', '18_c.png')
wavelet_decomposition('lionc.PNG')