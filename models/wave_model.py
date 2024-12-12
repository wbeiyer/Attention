import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import numpy as np
import os
from torchvision.utils import save_image
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

class WaveModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            # --lambda_L1 5
            parser.add_argument('--lambda_L1', type=float, default=50.0, help='weight for L1 loss')
            parser.add_argument('--lambda_freq', type=float, default=50.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the wave class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_freq','G_L1','D_real', 'D_fake']
        # todo
        self.loss_D_real = 0.0
        self.loss_D_fake = 0.0
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B','low_freq_fake','high_freq_fake','low_freq_real','high_freq_real']
        self.visual_names = ['real_A', 'fake_B', 'real_B','fake_B_LL','fake_B_LH','fake_B_HL','fake_B_HH','real_B_LL','real_B_LH','real_B_HL','real_B_HH']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    def wavelet_decomposition(self, x):
        pool = WavePool2(in_channels=3).to(x.device)
        # LL, LH, HL, HH = pool(x)
        return pool(x)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB是一个布尔值
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # 小波分解
        self.real_B_LL, self.real_B_LH, self.real_B_HL, self.real_B_HH = self.wavelet_decomposition(self.real_B)
        # save_image(self.real_B_LL, 'real_B_LL.png')
        # save_image(self.real_B_HL, 'real_B_HL.png')
        # save_image(self.real_B_LH, 'real_B_LH.png')
        # save_image(self.real_B_HH, 'real_B_HH.png')

        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    # self.visual_names = ['real_A', 'fake_B', 'real_B','low_freq_fake','low_freq_real','fake_B_LH','fake_B_HL','fake_B_HH','real_B_LH','real_B_HL','real_B_HH']
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B,self.fake_B_LL,self.high_freq_fake = self.netG(self.real_A)  # G(A)
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B_LL, self.fake_B_LH, self.fake_B_HL, self.fake_B_HH = self.wavelet_decomposition(self.fake_B)
    def calculate_loss_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach()) #detach() 方法断开梯度流，防止梯度传递到生成器
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    def backward_D(self):
        self.calculate_loss_D()
        self.loss_D.backward()

    def calculate_loss_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        #  self.loss_names = ['G_GAN', 'G_low_freq','G_LH','G_HL','G_HH','G_L1','D_real', 'D_fake']
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_LL= self.criterionL1(self.fake_B_LL,self.real_B_LL)
        self.loss_G_LH= self.criterionL1(self.fake_B_LH,self.real_B_LH)
        self.loss_G_HL= self.criterionL1(self.fake_B_HL,self.real_B_HL)
        self.loss_G_HH= self.criterionL1(self.fake_B_HH,self.real_B_HH)
        self.loss_G_freq=(self.loss_G_LL+self.loss_G_LH+self.loss_G_HL+self.loss_G_HH)* self.opt.lambda_freq
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_freq + self.loss_G_L1

    def backward_G(self):
        self.calculate_loss_G()
        self.loss_G.backward()

    def optimize_parameters(self):
        #核心代码
        # Calculate losses, gradients, and update network weights; called in every training iteration
        self.forward()                   # compute fake images: G(A)
        # update D 
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero 确保梯度不会累计到之前的梯度上
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
