import torch
import torch.nn as nn

def conv_3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def upconv_2x2(in_channels, out_channels):
    return  nn.Sequential(
        nn.Upsample(mode='bilinear', scale_factor=2,align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
    )


class UNet(nn.Module):

    def __init__(self,num_classes):
        super(UNet, self).__init__()

        self.dconv_1 = conv_3x3(1, 64)
        self.dconv_2 = conv_3x3(64, 128)
        self.dconv_3 = conv_3x3(128, 256)
        self.dconv_4 = conv_3x3(256, 512)
        self.dconv_5 = conv_3x3(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = upconv_2x2(1024, 512)
        self.dconv_up4 = conv_3x3(1024, 512)

        self.upconv3 = upconv_2x2(512, 256)
        self.dconv_up3 = conv_3x3(512, 256)

        self.upconv2 = upconv_2x2(256, 128)
        self.dconv_up2 = conv_3x3(256, 128)

        self.upconv1 = upconv_2x2(128, 64)
        self.dconv_up1 = conv_3x3(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        conv1d = self.dconv_1(x)
        x = self.maxpool(conv1d) 

        conv2d = self.dconv_2(x)
        x = self.maxpool(conv2d)
        
        conv3d = self.dconv_3(x)
        x = self.maxpool(conv3d)   
        
        conv4d = self.dconv_4(x)
        x = self.maxpool(conv4d)

        conv5d = self.dconv_5(x)

        x = self.upconv4(conv5d)
        conv4u = self.dconv_up4(torch.cat([x, conv4d], dim=1))

        x = self.upconv3(conv4u)
        conv3u = self.dconv_up3(torch.cat([x, conv3d], dim=1))

        x = self.upconv2(conv3u)
        conv2u = self.dconv_up2(torch.cat([x, conv2d], dim=1))

        x = self.upconv1(conv2u)
        conv1u = self.dconv_up1(torch.cat([x, conv1d], dim=1))

        # out = self.out(conv1u)
        out = self.out(conv1u)

        return out
