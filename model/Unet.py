import torch
import torch.nn as nn

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

class SingleConv_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block,self).__init__()
        self.conv3x3_1 = SingleConv(ch_in, ch_out//4)
        self.conv5x5_1 = SingleConv(ch_out//4, ch_out//4)
        self.conv7x7_1 = SingleConv(ch_out//4, ch_out//4)
        self.conv9x9_1 = SingleConv(ch_out//4, ch_out//4)
        self.short_cut = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_3x3_1 = self.conv3x3_1(x)
        x_5x5_1 = self.conv5x5_1(x_3x3_1)
        x_7x7_1 = self.conv7x7_1(x_5x5_1)
        x_9x9_1 = self.conv9x9_1(x_7x7_1)
        torch.cat((x_3x3_1, x_5x5_1, x_7x7_1, x_9x9_1), dim=1)
        torch.cat((x_3x3_1, x_5x5_1, x_7x7_1, x_9x9_1), dim=1).add_(self.short_cut(x))
        out = torch.add(torch.cat((x_3x3_1, x_5x5_1, x_7x7_1, x_9x9_1), dim=1), self.short_cut(x))
        out = self.relu(out)
        return out

# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block,self).__init__()
#         self.conv3x3_1 = SingleConv(ch_in, ch_out)
#         self.conv5x5_1 = SingleConv(ch_out, ch_out)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x_3x3_1 = self.conv3x3_1(x)
#         x_5x5_1 = self.conv5x5_1(x_3x3_1)
#         return x_5x5_1

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet(nn.Module):

    def __init__(self, n_class=3, features=[32, 64, 128], pool=[8, 4, 2]):
        super().__init__()
        self.dconv_down1 = conv_block(12, features[0])
        self.dconv_down2 = conv_block(features[0], features[1])
        self.dconv_down3 = conv_block(features[1], features[2])

        self.maxpool = nn.MaxPool2d(2)
        self.maxpool1 = nn.Conv2d(features[0], features[0], kernel_size=2, stride=2, padding=0)
        self.maxpool2 = nn.Conv2d(features[1], features[1], kernel_size=2, stride=2, padding=0)

        self.maxpool_spp_low = nn.MaxPool2d(pool[0], stride=pool[0])
        self.maxpool_spp_med1 = nn.MaxPool2d(pool[1], stride=pool[1])
        self.maxpool_spp_med2 = nn.MaxPool2d(pool[2], stride=pool[2])

        self.upsample_low = nn.Upsample(scale_factor=pool[0], mode='bilinear', align_corners=True)
        self.upsample_med1 = nn.Upsample(scale_factor=pool[1], mode='bilinear', align_corners=True)
        self.upsample_med2 = nn.Upsample(scale_factor=pool[2], mode='bilinear', align_corners=True)

        self.ori_compress = SingleConv_1x1(features[2], features[2]//4)
        self.low_compress = SingleConv_1x1(features[2], features[2]//4)
        self.med1_compress = SingleConv_1x1(features[2], features[2]//4)
        self.med2_compress = SingleConv_1x1(features[2], features[2]//4)

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.Att2 = Attention_block(F_g=features[1], F_l=features[1], F_int=features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.Att1 = Attention_block(F_g=features[0], F_l=features[0], F_int=features[0])

        self.dconv_up2 = conv_block(features[1] + features[1], features[1])
        self.dconv_up1 = conv_block(features[0] + features[0], features[0])
        self.conv_last = nn.Conv2d(features[0], n_class, kernel_size=1, padding=0, stride=1, bias=True)
        self.h_sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        ｘ = self.dconv_down3(x)

        spp_0 = self.ori_compress(x)

        spp_1 = self.maxpool_spp_low(ｘ)
        spp_1 = self.low_compress(spp_1)
        spp_1 = self.upsample_low(spp_1)

        spp_2 = self.maxpool_spp_med1(ｘ)
        spp_2 = self.med1_compress(spp_2)
        spp_2 = self.upsample_med1(spp_2)

        spp_3 = self.maxpool_spp_med2(ｘ)
        spp_3 = self.med1_compress(spp_3)
        spp_3 = self.upsample_med2(spp_3)

        x = self.up2(torch.cat([spp_1, spp_2, spp_3, spp_0], dim=1))
        #　x = self.up2(x)
        conv2 = self.Att2(g=x, x=conv2)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.up1(x)
        conv1 = self.Att1(g=x, x=conv1)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.h_sigmoid(self.conv_last(x))

        return out