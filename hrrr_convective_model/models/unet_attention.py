import torch, torch.nn as nn, torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g); x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_c, out_c),
            ConvBNReLU(out_c, out_c),
        )
    def forward(self, x, skip, attn):
        x = self.up(x)
        skip = attn(x, skip)  # Fixed: g=x (upsampled), x=skip
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetAttn(nn.Module):
    def __init__(self, in_ch, out_ch, nf=64):
        super().__init__()
        self.d1 = nn.Sequential(ConvBNReLU(in_ch, nf), ConvBNReLU(nf, nf))
        self.p1 = nn.MaxPool2d(2)
        self.d2 = nn.Sequential(ConvBNReLU(nf, nf*2), ConvBNReLU(nf*2, nf*2))
        self.p2 = nn.MaxPool2d(2)
        self.d3 = nn.Sequential(ConvBNReLU(nf*2, nf*4), ConvBNReLU(nf*4, nf*4))
        self.p3 = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(ConvBNReLU(nf*4, nf*8), ConvBNReLU(nf*8, nf*8))

        self.a3 = AttentionBlock(F_g=nf*4, F_l=nf*4, F_int=nf*2)
        self.a2 = AttentionBlock(F_g=nf*2, F_l=nf*2, F_int=nf)
        self.a1 = AttentionBlock(F_g=nf,   F_l=nf,   F_int=nf//2)

        self.u3 = UpBlock(nf*8+nf*4, nf*4)
        self.u2 = UpBlock(nf*4+nf*2, nf*2)
        self.u1 = UpBlock(nf*2+nf,   nf)

        self.outc = nn.Conv2d(nf, out_ch, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        b  = self.bridge(self.p3(d3))

        u3 = self.u3(b, d3, self.a3)
        u2 = self.u2(u3, d2, self.a2)
        u1 = self.u1(u2, d1, self.a1)
        return self.outc(u1)