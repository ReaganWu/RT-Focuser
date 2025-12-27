"""
RT-Focuser: Real-Time Lightweight Model for Edge-side Image Deblurring
Model architecture with paper-consistent naming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SN_Module(nn.Module):
    """
    结合锐化卷积和批归一化的可学习层
    输入形状: (N, C, H, W)
    输出形状: (N, C, H, W)
    """
    
    def __init__(self, channels, kernel_size=3, eps=1e-5, momentum=0.1, affine=True):
        """
        参数:
            channels: 输入通道数
            kernel_size: 锐化卷积核大小 (默认为3x3)
            eps: 批归一化的小常数
            momentum: 批归一化的动量
            affine: 是否启用可学习的缩放和偏移参数
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 初始化可学习的锐化卷积核 (每个通道独立)
        self.kernel = nn.Parameter(torch.zeros(channels, 1, kernel_size, kernel_size))
        
        # 初始化锐化核为拉普拉斯核
        laplacian = torch.tensor([[-1, -1, -1], 
                                  [-1,  8, -1], 
                                  [-1, -1, -1]], dtype=torch.float32)
        self.kernel.data = laplacian.repeat(channels, 1, 1, 1) / (kernel_size ** 2)
        
        # BatchNorm参数
        self.bn = nn.BatchNorm2d(channels, eps, momentum, affine)
        
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
        self.bn.reset_parameters()
        
    def forward(self, x):
        """
        前向传播:
        1. 应用锐化卷积
        2. 应用批归一化
        """
        # 应用深度可分离卷积实现锐化 (每个通道单独处理)
        sharpened = F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)
        
        # 应用批归一化
        return self.bn(sharpened)
    
    def extra_repr(self):
        return (f'channels={self.channels}, kernel_size={self.kernel_size}, '
                f'padding={self.padding}, affine={self.bn.affine}')


class Residual(nn.Module):
    def __init__(self, fn, ch_in=None):
        super().__init__()
        self.fn = fn
        self.ch_in = ch_in
        self.denoiser = SN_Module(channels=self.ch_in)
    def forward(self, x):
        return self.fn(x) + x + self.denoiser(x)


class LD_Block(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(LD_Block, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                ), ch_in=ch_in),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class MLIA(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        # 每个输入通道数通过 1x1 conv 统一映射到 out_channels
        self.branches = nn.ModuleList()
        for in_ch in in_channels_list:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 用于融合多个分支后再进行一次卷积提取
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # SE 注意力机制：通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 输出 [B, C, 1, 1]
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        inputs: List of tensors, each is [B, C_i, H, W]
        """

        # Step 1: 先对每个输入通过 1x1 conv 统一维度
        projected_feats = []
        for feat, branch in zip(inputs, self.branches):
            projected_feats.append(branch(feat))  # 每个结果形状都是 [B, out_channels, H, W]

        # Step 2: concat 后融合
        concat_feats = torch.cat(projected_feats, dim=1)  # [B, out_channels * N, H, W]
        fused = self.fusion_conv(concat_feats)  # [B, out_channels, H, W]

        # Step 3: SE模块，生成通道注意力
        channel_weights = self.channel_attention(fused)  # [B, out_channels, 1, 1]
        out = fused * channel_weights  # 通道加权

        return out


class XFuse_Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(XFuse_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out+3, ch_out, kernel_size=3, stride=1, padding=1, groups=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x, skip):
        # print("x shape:", x.shape, "skip shape:", skip.shape)
        x = self.conv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv2(x)
        return x


class RT_Focuser(nn.Module):
    def __init__(self, input_channel=3, dims=[16, 32, 128, 160, 256], depths=[3, 3, 3, 3, 2], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(RT_Focuser, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = LD_Block(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = LD_Block(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = LD_Block(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = LD_Block(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = LD_Block(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = XFuse_Block(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = XFuse_Block(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = XFuse_Block(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = XFuse_Block(ch_in=dims[0] * 2, ch_out=dims[0])
        # self.Up_conv2 = fusion_conv_deblur(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], 3, kernel_size=1, stride=1, padding=0)

        self.Msf_8 = MLIA([dims[0], dims[1], dims[2], dims[3]], dims[3])
        self.Msf_4 = MLIA([dims[0], dims[1], dims[2], dims[3]], dims[2])
        self.Msf_2 = MLIA([dims[0], dims[1], dims[2], dims[3]], dims[1])
        self.Msf_1 = MLIA([dims[0], dims[1], dims[2], dims[3]], dims[0])
        # self.Msf_1 = MSF_SE([dims[0], dims[1], dims[2], dims[3]], dims[0])
        
    def forward(self, x):
        o1 = x
        o2 = F.interpolate(o1, scale_factor=0.5)    # H, W -> 2/H, 2/W
        o3 = F.interpolate(o2, scale_factor=0.5)   # H, W -> 4/H, 4/o 
        o4 = F.interpolate(o3, scale_factor=0.5)  # H, W -> 8/H, 8/W

        x1 = self.stem(x)           # x1 shape: [B, 16, H, W]
        x1 = self.encoder1(x1)      # x1 shape: [B, 16, H, W]
        x2 = self.Maxpool(x1)       # x2 shape: [B, 16, H/2, W/2]
        x2 = self.encoder2(x2)      # x2 shape: [B, 32, H/2, W/2]
        x3 = self.Maxpool(x2)       # x3 shape: [B, 32, H/4, W/4]
        x3 = self.encoder3(x3)      # x3 shape: [B, 128, H/4, W/4]
        x4 = self.Maxpool(x3)       # x4 shape: [B, 128, H/8, W/8]
        x4 = self.encoder4(x4)      # x4 shape: [B, 160, H/8, W/8]
        x5 = self.Maxpool(x4)       # x5 shape: [B, 160, H/16, W/16]
        x5 = self.encoder5(x5)      # x5 shape: [B, 256, H/16, W/16]

        # wanna make x1 downsample to x4, x3, x2
        x12 = F.interpolate(x1, scale_factor=0.5)  # x12 shape: [B, 16, H/2, W/2]
        x13 = F.interpolate(x12, scale_factor=0.5) # x13 shape: [B, 16, H/4, W/4]
        x14 = F.interpolate(x13, scale_factor=0.5) # x14 shape: [B, 16, H/8, W/8]
        # wanna make x2 downsample to x4, x3
        x21 = F.interpolate(x2, scale_factor=2)  # x21 shape: [B, 32, H, W]
        x23 = F.interpolate(x2, scale_factor=0.5)  # x22 shape: [B, 32, H/4, W/4]
        x24 = F.interpolate(x23, scale_factor=0.5) # x23 shape: [B, 32, H/8, W/8]
        # wanna make x3 downsample to x4
        x34 = F.interpolate(x3, scale_factor=0.5)  # x32 shape: [B, 128, H/8, W/8]
        x32 = F.interpolate(x3, scale_factor=2)  # x32 shape: [B, 128, H/2, W/2]
        x31 = F.interpolate(x32, scale_factor=2)  # x31 shape: [B, 128, H, W]
        # wanna make x4 upsample to 
        x43 = F.interpolate(x4, scale_factor=2)  # x43 shape: [B, 160, H/4, W/4]
        x42 = F.interpolate(x43, scale_factor=2)  # x42 shape: [B, 160, H/2, W/2]
        x41 = F.interpolate(x42, scale_factor=2)  # x41 shape: [B, 160, H, W]

        # H/8: x14, x24, x34, x4
        # H/4: x13, x23, x43, x3
        # H/2: x12, x32, x42, x2
        # H:   x21, x31, x41, x1
        # fusion H/8 -> replace x4 by f4, x4 shape: [B, 160, H/8, W/8]
        # fusion H/4 -> replace x3 by f3, x3 shape: [B, 128, H/4, W/4]
        # fusion H/2 -> replace x2 by f2, x2 shape: [B, 32, H/2, W/2]
        f4 = self.Msf_8([x14, x24, x34, x4])  # f4 shape: [B, 16+32+128+160, H/8, W/8]  = [B, 336, H/8, W/8], x4 shape: [B, 160, H/8, W/8]
        f3 = self.Msf_4([x13, x23, x3, x43])  # f3 shape: [B, 16+32+128+160, H/4, W/4] = [B, 336, H/4, W/4], x3 shape: [B, 128, H/4, W/4]
        f2 = self.Msf_2([x12, x2,  x32, x42])  # f2 shape: [B, 16+32+128+160, H/2, W/2] = [B, 336, H/2, W/2], x2 shape: [B, 32, H/2, W/2]
        f1 = self.Msf_1([x1, x21, x31, x41])  # f1 shape: [B, 16+32+128+160, H, W] = [B, 336, H, W], x1 shape: [B, 16, H, W]

        d5 = self.Up5(x5)           # d5 shape: [B, 160, H/8, W/8]
        # d5 = torch.cat((x4, d5), dim=1) # d5 shape: [B, 320, H/8, W/8]
        d5 = torch.cat((f4, d5), dim=1) # d5 shape: [B, 336, H/8, W/8]
        d5 = self.Up_conv5(d5, o4)      # d5 shape: [B, 160, H/8, W/8]

        d4 = self.Up4(d5)           # d4 shape: [B, 128, H/4, W/4]
        # d4 = torch.cat((x3, d4), dim=1) # d4 shape: [B, 256, H/4, W/4]
        d4 = torch.cat((f3, d4), dim=1) # d4 shape: [B, 336, H/4, W/4]
        d4 = self.Up_conv4(d4, o3)     # d4 shape: [B, 128, H/4, W/4]

        d3 = self.Up3(d4)           # d3 shape: [B, 32, H/2, W/2]
        # d3 = torch.cat((x2, d3), dim=1)     # d3 shape: [B, 64, H/2, W/2]
        d3 = torch.cat((f2, d3), dim=1)   # d3 shape: [B, 336, H/2, W/2]
        d3 = self.Up_conv3(d3, o2)     # d3 shape: [B, 32, H/2, W/2]

        d2 = self.Up2(d3)           # d2 shape: [B, 16, H, W]
        # d2 = torch.cat((x1, d2), dim=1)   # d2 shape: [B, 32, H, W]
        d2 = torch.cat((f1, d2), dim=1)  # d2 shape: [B, 336, H, W]
        d2 = self.Up_conv2(d2, o1)  # d2 shape: [B, 16, H, W]
        
        d1 = self.Conv_1x1(d2)      # d1 shape: [B, 3, H, W]
        d1 = d1 + o1  # attemp 1: add o1 before d1
        d1 = torch.sigmoid(d1)
        # d1 = d1 + o1  # attemp 0: add o1 after d1
        # print("d1 shape:", d1.shape)
        
        return d1


def RT_Focuser_Standard(dims=[16, 32, 128, 160, 256], depths=[3, 4, 4, 3, 2], kernels=[3, 3, 7, 7, 7]):
    return RT_Focuser(dims=dims, depths=depths, kernels=kernels)

if __name__ == "__main__":
    model = RT_Focuser_Standard()
    # Example usage:
    input_tensor = torch.randn(1, 3, 256, 256)  # Example input tensor
    output = model(input_tensor)
    print('output shape:', output.shape)  # Should print torch.Size([1, 3, 256, 256])
    print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # load pretrained weights
    model_path = "/Users/reaganwu/Documents/GitHub/RT-Focuser/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)

    # import ptflops, you should install it, if you wanna check the macs
    # macs, params = ptflops.get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    # print(f'MACs: {macs}, Params: {params}')
