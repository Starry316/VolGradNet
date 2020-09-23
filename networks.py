"""
Network structure definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_modules import ConvBlock, DeConvBlock, ResidualBlock


class VolGradNet(nn.Module):
    def __init__(self, feature_channel=12):
        super(VolGradNet, self).__init__()
        self.grad_conv_1 = ConvBlock(2, 32, stride=1)
        self.grad_conv_2 = ConvBlock(32, 32, stride=1)

        self.img_conv_1 = ConvBlock(1, 32, stride=1)
        self.img_conv_2 = ConvBlock(32, 32, stride=1)

        self.feature_conv_1 = ConvBlock(feature_channel, 32, stride=1)
        self.feature_conv_2 = ConvBlock(32, 32, stride=1)

        self.residual_blocks_esc = nn.Sequential(
            ResidualBlock(96),
            ResidualBlock(96),
        )
        self.grad_down_1 = ConvBlock(2, 32)
        self.grad_down_2 = ConvBlock(32, 64)
        self.grad_down_3 = ConvBlock(64, 128)

        self.img_down_1 = ConvBlock(1, 32)
        self.img_down_2 = ConvBlock(32, 64)
        self.img_down_3 = ConvBlock(64, 128)

        self.feature_down_1 = ConvBlock(feature_channel, 32)
        self.feature_down_2 = ConvBlock(32, 64)
        self.feature_down_3 = ConvBlock(64, 128)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(384),
            ResidualBlock(384),
            ResidualBlock(384),
            ResidualBlock(384)
        )

        self.up_1 = DeConvBlock(768, 192)
        self.up_2 = DeConvBlock(384, 96)
        self.up_3 = DeConvBlock(192, 32)

        self.all_conv = ConvBlock(128, 32, stride=1)

        self.final_process = nn.Sequential(
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def ESC(self, i, feature, grad):
        g_1 = self.grad_conv_1(grad)
        g_2 = self.grad_conv_2(g_1)

        i_1 = self.img_conv_1(i)
        i_2 = self.img_conv_2(i_1)

        f_1 = self.feature_conv_1(feature)
        f_2 = self.feature_conv_2(f_1)

        i_c = torch.cat([i_2, g_2, f_2], 1)
        return self.residual_blocks_esc(i_c)

    def UNet(self, i, feature, grad):
        g_down_1 = self.grad_down_1(grad)
        g_down_2 = self.grad_down_2(g_down_1)
        g_down_3 = self.grad_down_3(g_down_2)

        i_down_1 = self.img_down_1(i)
        i_down_2 = self.img_down_2(i_down_1)
        i_down_3 = self.img_down_3(i_down_2)

        f_down_1 = self.feature_down_1(feature)
        f_down_2 = self.feature_down_2(f_down_1)
        f_down_3 = self.feature_down_3(f_down_2)

        i_cat = torch.cat([i_down_3, g_down_3, f_down_3], 1)
        i_res = self.residual_blocks(i_cat)

        i_up_1 = self.up_1(torch.cat([i_res, i_down_3, g_down_3, f_down_3], 1))
        i_up_2 = self.up_2(torch.cat([i_up_1, i_down_2, g_down_2, f_down_2], 1))
        return self.up_3(torch.cat([i_up_2, i_down_1, g_down_1, f_down_1], 1))

    def forward(self, i, feature, grad):
        res_esc = self.ESC(i, feature, grad)
        res_unet = self.UNet(i, feature, grad)
        all = self.all_conv(torch.cat([res_esc, res_unet], 1))
        return self.final_process(all)


class GradNet(nn.Module):
    """
    GradNet
    """
    def __init__(self, feature_channel=7):
        super(GradNet, self).__init__()

        self.grad_down_1 = ConvBlock(2, 32)
        self.grad_down_2 = ConvBlock(32, 64)
        self.grad_down_3 = ConvBlock(64, 128)

        self.img_down_1 = ConvBlock(1 + feature_channel, 32)
        self.img_down_2 = ConvBlock(32, 64)
        self.img_down_3 = ConvBlock(64, 128)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.up_1 = DeConvBlock(512, 128)
        self.up_2 = DeConvBlock(256, 64)
        self.up_3 = DeConvBlock(128, 32)

        self.final_process = nn.Sequential(
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, i, feature, grad):
        g_down_1 = self.grad_down_1(grad)
        g_down_2 = self.grad_down_2(g_down_1)
        g_down_3 = self.grad_down_3(g_down_2)

        i_down_1 = self.img_down_1(torch.cat([i, feature], 1))
        i_down_2 = self.img_down_2(i_down_1)
        i_down_3 = self.img_down_3(i_down_2)

        i_cat = torch.cat([i_down_3, g_down_3], 1)
        i_res = self.residual_blocks(i_cat)
        i_up_1 = self.up_1(torch.cat([i_res, i_down_3, g_down_3], 1))
        i_up_2 = self.up_2(torch.cat([i_up_1, i_down_2, g_down_2], 1))
        i_up_3 = self.up_3(torch.cat([i_up_2, i_down_1, g_down_1], 1))

        i_ret = self.final_process(i_up_3)
        return i_ret


class GradGenNet(nn.Module):
    """
    G-branch
    """
    def __init__(self, feature_channel=7):
        super(GradGenNet, self).__init__()

        self.conv_1 = ConvBlock(3 + feature_channel, 64, stride=1)
        self.conv_2 = ConvBlock(64, 64, stride=1)
        self.conv_3 = nn.Conv2d(64, feature_channel, 3, stride=1, padding=1)

    def forward(self, i):
        i = self.conv_1(i)
        i = self.conv_2(i)
        i = self.conv_3(i)
        return i


class PUnit(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 2 * out_channel, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * out_channel, out_channel, 3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        return self.lrelu(x)


class NGPT(nn.Module):
    def __init__(self):
        super(NGPT, self).__init__()

        self.fullRes1 = PUnit(10, 40)
        self.fullRes2 = PUnit(10 + 40, 40)
        self.fullRes3 = PUnit(10 + 80, 40)
        self.fullRes4 = PUnit(10 + 120, 40)
        self.down1 = nn.Conv2d(10 + 160, 160, 2, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.halfRes1 = PUnit(160, 80)
        self.halfRes2 = PUnit(160 + 80, 80)
        self.halfRes3 = PUnit(160 + 160, 80)
        self.down2 = nn.Conv2d(160 + 240, 160, 2, stride=1, padding=1)

        self.quartRes1 = PUnit(160, 80)
        self.quartRes2 = PUnit(160 + 80, 80)
        self.down3 = nn.Conv2d(160 + 160, 160, 2, stride=1, padding=1)

        self.eightRes1 = PUnit(160, 80)
        self.eightRes2 = PUnit(160 + 80, 80)
        self.eightRes3 = PUnit(160 + 160, 80)
        self.eightRes4 = PUnit(160 + 240, 80)
        self.up1 = nn.ConvTranspose2d(160 + 320, 160, 4, stride=2, padding=1)

        self.quartRes3 = PUnit(160 + 160 + 160, 80)
        self.quartRes4 = PUnit(160 + 160 + 160 + 80, 80)
        self.up2 = nn.ConvTranspose2d(160 + 160 + 160 + 160, 160, 4, stride=2, padding=1)

        self.halfRes4 = PUnit(160 + 240 + 160, 80)
        self.halfRes5 = PUnit(160 + 240 + 160 + 80, 80)
        self.halfRes6 = PUnit(160 + 240 + 160 + 160, 80)
        self.up3 = nn.ConvTranspose2d(160 + 240 + 160 + 240, 80, 4, stride=2, padding=1)

        self.fullRes5 = PUnit(10 + 160 + 80, 40)
        self.fullRes6 = PUnit(10 + 160 + 80 + 40, 40)
        self.fullRes7 = PUnit(10 + 160 + 80 + 80, 40)
        self.fullRes8 = PUnit(10 + 160 + 80 + 120, 40)
        self.fin = nn.Conv2d(10 + 160 + 80 + 160, 1, 1, stride=1, padding=0)

    def forward(self, i,feature, grad):
        i = torch.cat([i, grad,feature],1)
        f1 = F.leaky_relu(self.fullRes1(i))
        f1 = torch.cat([i, f1], 1)
        t = F.leaky_relu(self.fullRes2(f1))
        f1 = torch.cat([t, f1], 1)
        t = F.leaky_relu(self.fullRes3(f1))
        f1 = torch.cat([t, f1], 1)
        t = F.leaky_relu(self.fullRes4(f1))
        f1 = torch.cat([t, f1], 1)
        h1 =  self.pool(F.leaky_relu(self.down1(f1)))
        t = F.leaky_relu(self.halfRes1(h1))
        h1 = torch.cat([t, h1], 1)
        t = F.leaky_relu(self.halfRes2(h1))
        h1 = torch.cat([t, h1], 1)
        t = F.leaky_relu(self.halfRes3(h1))
        h1 = torch.cat([t, h1], 1)
        q1 = self.pool(F.leaky_relu(self.down2(h1)))
        t = F.leaky_relu(self.quartRes1(q1))
        q1 = torch.cat([t, q1], 1)
        t = F.leaky_relu(self.quartRes2(q1))
        q1 = torch.cat([t, q1], 1)
        e1 = self.pool(F.leaky_relu(self.down3(q1)))
        t = F.leaky_relu(self.eightRes1(e1))
        e1 = torch.cat([t, e1], 1)
        t = F.leaky_relu(self.eightRes2(e1))
        e1 = torch.cat([t, e1], 1)
        t = F.leaky_relu(self.eightRes3(e1))
        e1 = torch.cat([t, e1], 1)
        t = F.leaky_relu(self.eightRes4(e1))
        e1 = torch.cat([t, e1], 1)
        # print(e1.shape)
        t =  F.leaky_relu(self.up1(e1))
        # print(t.shape)
        q1 = torch.cat([t, q1], 1)
        t =  F.leaky_relu(self.quartRes3(q1))
        q1 = torch.cat([t, q1], 1)
        t =  F.leaky_relu(self.quartRes4(q1))
        q1 = torch.cat([t, q1], 1)
        t =  F.leaky_relu(self.up2(q1))
        h1 = torch.cat([t, h1], 1)
        t =  F.leaky_relu(self.halfRes4(h1))
        h1 = torch.cat([t, h1], 1)
        t = F.leaky_relu(self.halfRes5(h1))
        h1 = torch.cat([t, h1], 1)
        t = F.leaky_relu(self.halfRes6(h1))
        h1 = torch.cat([t, h1], 1)
        t = F.leaky_relu(self.up3(h1))
        f1 = torch.cat([t, f1], 1)
        t = F.leaky_relu(self.fullRes5(f1))
        f1 = torch.cat([t, f1], 1)
        t = F.leaky_relu(self.fullRes6(f1))
        f1 = torch.cat([t, f1], 1)
        t = F.leaky_relu(self.fullRes7(f1))
        f1 = torch.cat([t, f1], 1)
        t = F.leaky_relu(self.fullRes8(f1))
        f1 = torch.cat([t, f1], 1)
        return F.leaky_relu(self.fin(f1))


class VolGradNet_NoESC(nn.Module):
    def __init__(self, feature_channel = 7):
        super(VolGradNet_NoESC, self).__init__()
        self.grad_down_1 = ConvBlock(2, 32)
        self.grad_down_2 = ConvBlock(32, 64)
        self.grad_down_3 = ConvBlock(64, 128)

        self.img_down_1 = ConvBlock(1, 32)
        self.img_down_2 = ConvBlock(32, 64)
        self.img_down_3 = ConvBlock(64, 128)

        self.feature_down_1 = ConvBlock(feature_channel, 32)
        self.feature_down_2 = ConvBlock(32, 64)
        self.feature_down_3 = ConvBlock(64, 128)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(384),
            ResidualBlock(384),
            ResidualBlock(384),
            ResidualBlock(384)
        )

        self.up_1 = DeConvBlock(768, 192)
        self.up_2 = DeConvBlock(384, 96)
        self.up_3 = DeConvBlock(192, 32)


        self.final_process = nn.Sequential(
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.ReLU()
        )


    def UNet(self, i, feature, grad):
        g_down_1 = self.grad_down_1(grad)
        g_down_2 = self.grad_down_2(g_down_1)
        g_down_3 = self.grad_down_3(g_down_2)

        i_down_1 = self.img_down_1(i)
        i_down_2 = self.img_down_2(i_down_1)
        i_down_3 = self.img_down_3(i_down_2)

        f_down_1 = self.feature_down_1(feature)
        f_down_2 = self.feature_down_2(f_down_1)
        f_down_3 = self.feature_down_3(f_down_2)

        i_cat = torch.cat([i_down_3, g_down_3, f_down_3], 1)
        i_res = self.residual_blocks(i_cat)

        i_up_1 = self.up_1(torch.cat([i_res, i_down_3, g_down_3, f_down_3], 1))
        i_up_2 = self.up_2(torch.cat([i_up_1, i_down_2, g_down_2, f_down_2], 1))
        return self.up_3(torch.cat([i_up_2, i_down_1, g_down_1, f_down_1], 1))

    def forward(self, i, feature, grad):
        res_unet = self.UNet(i, feature, grad)
        return self.final_process(res_unet)


class VolGradNet_NoFB(nn.Module):
    def __init__(self, feature_channel = 7):
        super(VolGradNet_NoFB, self).__init__()
        self.grad_conv_1 = ConvBlock(2, 32, stride=1)
        self.grad_conv_2 = ConvBlock(32, 32, stride=1)

        self.img_conv_1 = ConvBlock(1 + feature_channel, 32, stride=1)
        self.img_conv_2 = ConvBlock(32, 32, stride=1)


        self.residual_blocks_esc = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )
        self.grad_down_1 = ConvBlock(2, 32)
        self.grad_down_2 = ConvBlock(32, 64)
        self.grad_down_3 = ConvBlock(64, 128)

        self.img_down_1 = ConvBlock(1 + feature_channel, 32)
        self.img_down_2 = ConvBlock(32, 64)
        self.img_down_3 = ConvBlock(64, 128)


        self.residual_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.up_1 = DeConvBlock(512, 128)
        self.up_2 = DeConvBlock(256, 64)
        self.up_3 = DeConvBlock(128, 32)

        self.all_conv = ConvBlock(96, 32, stride=1)

        self.final_process = nn.Sequential(
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def ESC(self, i, feature, grad):
        g_1 = self.grad_conv_1(grad)
        g_2 = self.grad_conv_2(g_1)

        i_1 = self.img_conv_1(torch.cat([i, feature], 1))
        i_2 = self.img_conv_2(i_1)

        i_c = torch.cat([i_2, g_2], 1)
        return self.residual_blocks_esc(i_c)

    def UNet(self, i, feature, grad):
        g_down_1 = self.grad_down_1(grad)
        g_down_2 = self.grad_down_2(g_down_1)
        g_down_3 = self.grad_down_3(g_down_2)

        i_down_1 = self.img_down_1(torch.cat([i, feature], 1))
        i_down_2 = self.img_down_2(i_down_1)
        i_down_3 = self.img_down_3(i_down_2)

        i_cat = torch.cat([i_down_3, g_down_3], 1)
        i_res = self.residual_blocks(i_cat)

        i_up_1 = self.up_1(torch.cat([i_res, i_down_3, g_down_3], 1))
        i_up_2 = self.up_2(torch.cat([i_up_1, i_down_2, g_down_2], 1))
        return self.up_3(torch.cat([i_up_2, i_down_1, g_down_1], 1))

    def forward(self, i, feature, grad):
        res_esc = self.ESC(i, feature, grad)
        res_unet = self.UNet(i, feature, grad)
        all = self.all_conv(torch.cat([res_esc, res_unet], 1))
        return self.final_process(all)



