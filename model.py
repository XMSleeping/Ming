import torch
import torch.nn as nn
import torch.nn.functional as F


# CBAM 注意力模块
class CBAM(nn.Module):
    """通道注意力 + 空间注意力"""
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.channels = channels
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        
        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # 通道注意力
        avg_out = self.mlp(self.avg_pool(x).view(B, C))
        max_out = self.mlp(self.max_pool(x).view(B, C))
        channel_attn = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        x = x * channel_attn
        
        # 空间注意力
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        x = x * spatial_attn
        
        return x

# 2. 基础卷积块
class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# U-Net + CBAM 
class UNet_CBAM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.skip_cbs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        # 编码器
        ch = in_channels
        for f in features:
            self.encoder_blocks.append(DoubleConv(ch, f))
            ch = f
        
        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # 解码器
        rev_features = list(reversed(features))
        for i, f in enumerate(rev_features):
            self.upsamplers.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.skip_cbs.append(CBAM(f))
            self.decoder_blocks.append(DoubleConv(f * 2, f))
        
        # 输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # 编码路径
        for enc in self.encoder_blocks:
            x = enc(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码路径
        skip_connections = skip_connections[::-1]
        
        for i in range(len(self.decoder_blocks)):
            # 上采样
            x = self.upsamplers[i](x)
            
            # 获取对应的跳跃连接
            skip = skip_connections[i]
            
            # 处理尺寸不匹配
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            
            # 对跳跃连接应用CBAM
            skip = self.skip_cbs[i](skip)
            
            # 拼接
            x = torch.cat([skip, x], dim=1)
            
            # 双重卷积
            x = self.decoder_blocks[i](x)
        
        return self.final_conv(x)

class SelfSupervisedUNet(nn.Module):
    """带自监督头的UNet"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], 
                 pretext_task='rotation'):
        super().__init__()
        self.pretext_task = pretext_task
        self.backbone = UNet_CBAM(in_channels, out_channels, features)
        
        # 冻结backbone的前几层（可选）
        # for param in list(self.backbone.parameters())[:-10]:
        #     param.requires_grad = False
        
        # 自监督头 - 修复输入维度
        if pretext_task == 'rotation':
            # 旋转预测：4个类别
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(features[-1] * 2, 256),  # 应该是 512 * 2 = 1024
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 4)
            )
        elif pretext_task == 'jigsaw':
            # 拼图任务：24种排列组合
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(features[-1] * 2, 256),  # 应该是 512 * 2 = 1024
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 24)
            )
        else:
            # 重建任务：使用原始UNet的输出
            self.head = None
    
    def forward(self, x):
        skip_connections = []
        
        # 编码路径
        for enc in self.backbone.encoder_blocks:
            x = enc(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 瓶颈层 
        features = self.backbone.bottleneck(x)
        
        if self.pretext_task in ['rotation', 'jigsaw']:
            # 自监督分类任务
            return self.head(features)
        else:
            x = features
            skip_connections = skip_connections[::-1]
            
            for i in range(len(self.backbone.decoder_blocks)):
                x = self.backbone.upsamplers[i](x)
                skip = skip_connections[i]
                
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
                
                skip = self.backbone.skip_cbs[i](skip)
                x = torch.cat([skip, x], dim=1)
                x = self.backbone.decoder_blocks[i](x)
            
            return self.backbone.final_conv(x)

# LMBiS-Net 多路径块
class MultipathBlock(nn.Module):
    """LMBiS-Net的多路径特征提取块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        
        # 分配通道数
        c1 = max(1, out_channels // 4)
        c3 = max(1, out_channels // 2)
        c5 = out_channels - c1 - c3
        
        # 三个并行路径
        self.path_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        
        self.path_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, c3, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        
        self.path_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, c5, kernel_size=5, padding=2, stride=stride, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )
        
        # 融合层
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # 残差连接
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        p1 = self.path_1x1(x)
        p3 = self.path_3x3(x)
        p5 = self.path_5x5(x)
        
        out = torch.cat([p1, p3, p5], dim=1)
        out = self.fuse(out)
        
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        else:
            out = out + x
            
        return out

# MBiS-Net 主模型
class LMBiSNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__()
        bf = base_filters
        
        # ---------- Stem层 ----------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, bf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bf),
            nn.ReLU(inplace=True),
        )
        
        # ---------- 编码器（----------
        # Stage 1: 512x512 -> 512x512
        self.enc1 = MultipathBlock(bf, bf * 2, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Stage 2: 256x256 -> 256x256
        self.enc2 = MultipathBlock(bf * 2, bf * 4, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Stage 3: 128x128 -> 128x128
        self.enc3 = MultipathBlock(bf * 4, bf * 8, stride=1)
        
        # ---------- 瓶颈层 ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bf * 8, bf * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bf * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(bf * 8, bf * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bf * 8),
            nn.ReLU(inplace=True),
        )
        
        # ---------- 解码器 + 正向CBAM ----------
        # Decoder 1: 128x128 -> 256x256
        self.up1 = nn.ConvTranspose2d(bf * 8, bf * 4, kernel_size=2, stride=2)
        self.cbam_skip2 = CBAM(bf * 4)  # 对应enc2的跳跃连接
        self.dec1 = MultipathBlock(bf * 8, bf * 4)  # 输入通道：bf*4(up) + bf*4(skip) = bf*8
        
        # Decoder 2: 256x256 -> 512x512
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, kernel_size=2, stride=2)
        self.cbam_skip1 = CBAM(bf * 2)  # 对应enc1的跳跃连接
        self.dec2 = MultipathBlock(bf * 4, bf * 2)  # 输入通道：bf*2(up) + bf*2(skip) = bf*4
        
        # ---------- 反向连接（门控信号）----------
        # 从dec1生成门控信号，作用于enc2
        self.reverse_gate2 = nn.Sequential(
            nn.Conv2d(bf * 4, bf * 4, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        
        # 从dec2生成门控信号，作用于enc1
        self.reverse_gate1 = nn.Sequential(
            nn.Conv2d(bf * 2, bf * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        
        # ---------- 最终精修层 ----------
        self.refine = nn.Sequential(
            nn.Conv2d(bf * 2 + bf, bf * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bf * 2, bf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bf),
            nn.ReLU(inplace=True),
        )
        
        # ---------- 输出层 ----------
        self.head = nn.Conv2d(bf, out_channels, kernel_size=1)

    def forward(self, x):
        # ---------- Stem ----------
        stem_out = self.stem(x)  # (bf, 512, 512)
        
        # ---------- 编码器 ----------
        enc1_out = self.enc1(stem_out)  # (bf*2, 512, 512)
        pool1_out = self.pool1(enc1_out)  # (bf*2, 256, 256)
        
        enc2_out = self.enc2(pool1_out)  # (bf*4, 256, 256)
        pool2_out = self.pool2(enc2_out)  # (bf*4, 128, 128)
        
        enc3_out = self.enc3(pool2_out)  # (bf*8, 128, 128)
        
        # ---------- 瓶颈层 ----------
        bottleneck_out = self.bottleneck(enc3_out)  # (bf*8, 128, 128)
        
        # ---------- 解码器 + 正向CBAM ----------
        up1_out = self.up1(bottleneck_out)  # (bf*4, 256, 256)
        skip2 = self.cbam_skip2(enc2_out)  # CBAM作用于enc2
        dec1_in = torch.cat([up1_out, skip2], dim=1)  # (bf*8, 256, 256)
        dec1_out = self.dec1(dec1_in)  # (bf*4, 256, 256)
        
        # 反向连接：dec1_out 作为门控信号作用于 enc2_out
        gate2 = self.reverse_gate2(dec1_out)  # (bf*4, 256, 256)
        enc2_gated = enc2_out * gate2  # 门控后的enc2特征
        
        up2_out = self.up2(dec1_out)  # (bf*2, 512, 512)
        skip1 = self.cbam_skip1(enc1_out)  # CBAM作用于enc1
        dec2_in = torch.cat([up2_out, skip1], dim=1)  # (bf*4, 512, 512)
        dec2_out = self.dec2(dec2_in)  # (bf*2, 512, 512)
        
        # 反向连接：dec2_out 作为门控信号作用于 enc1_out
        gate1 = self.reverse_gate1(dec2_out)  # (bf*2, 512, 512)
        enc1_gated = enc1_out * gate1  # 门控后的enc1特征
        
        # ---------- 精修层 ----------
        # 将dec2_out与stem_out拼接，保留低层细节
        refine_in = torch.cat([dec2_out, stem_out], dim=1)  # (bf*2+bf, 512, 512)
        refine_out = self.refine(refine_in)  # (bf, 512, 512)
        
        # ---------- 输出 ----------
        output = self.head(refine_out)  # (out_channels, 512, 512)
        
        return output

# 模型函数
def get_model(name: str, **kwargs):
    """
    获取模型实例
    
    参数:
        name: 模型名称 ('unet_cbam' 或 'lmbisnet')
        **kwargs: 模型参数
    
    返回:
        模型实例
    """
    name = name.lower().strip()
    
    if name in ['unet', 'unet_cbam', 'unet-cbam']:
        return UNet_CBAM(
            in_channels=kwargs.get('in_channels', 1),
            out_channels=kwargs.get('out_channels', 1),
            features=kwargs.get('features', [64, 128, 256, 512]),
        )
    
    elif name in ['lmbis', 'lmbisnet', 'lmbi', 'lmbsinet']:
        return LMBiSNet(
            in_channels=kwargs.get('in_channels', 1),
            out_channels=kwargs.get('out_channels', 1),
            base_filters=kwargs.get('base_filters', 16),
        )
    
    else:
        raise ValueError(f"未知模型名称: {name}. 请选择 'unet_cbam' 或 'lmbisnet'")