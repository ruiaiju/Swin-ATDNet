import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # 包含 Swin Transformer 的实现

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H/4, W/4]
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.norm(x)
        return x

# 空洞卷积模块
class AtrousConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, dilation=1, padding=0)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=3, padding=3)
        self.conv5 = nn.Conv2d(in_ch, out_ch, kernel_size=5, dilation=1, padding=2)

    def forward(self, x):
        return self.conv1(x) + self.conv3(x) + self.conv5(x)

# DP-HSAN 注意力模块
class DP_HSAN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.pool1(x)
        max_ = self.pool2(x)
        out = self.conv(avg + max_)
        return x * self.sigmoid(out)

# 解码器块
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attn = DP_HSAN(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.attn(x)

# 主模型
class SwinSegNet(nn.Module):
    def __init__(self, img_size=512, num_classes=1):
        super().__init__()
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True, features_only=True)
        self.atrous1 = AtrousConv(128, 128)
        self.atrous2 = AtrousConv(256, 256)
        self.atrous3 = AtrousConv(512, 512)

        self.decode1 = DecoderBlock(1024, 512)
        self.decode2 = DecoderBlock(512, 256)
        self.decode3 = DecoderBlock(256, 128)
        self.decode4 = DecoderBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.swin(x)  # List of features from swin: [C1, C2, C3, C4]
        c1, c2, c3, c4 = feats

        c1 = self.atrous1(c1)
        c2 = self.atrous2(c2)
        c3 = self.atrous3(c3)

        d1 = self.decode1(c4, c3)
        d2 = self.decode2(d1, c2)
        d3 = self.decode3(d2, c1)
        d4 = self.decode4(d3, x[:, :64, :, :])  # 初始skip连接

        return torch.sigmoid(self.final(d4))
