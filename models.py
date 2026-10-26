"""Pix2Pix-style conditional GAN: U-Net generator + 70x70 PatchGAN discriminator."""
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d) and m.affine:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(cout, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, cin, cout, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.InstanceNorm2d(cout, affine=True)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """6-level U-Net with skip connections. Input H, W must be multiples of 64 (and >= 128).

    Input/output range is [-1, 1].
    """

    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        c = [ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 8, ngf * 8]
        self.downs = nn.ModuleList([
            Down(in_ch, c[0], norm=False),
            Down(c[0], c[1]),
            Down(c[1], c[2]),
            Down(c[2], c[3]),
            Down(c[3], c[4]),
            Down(c[4], c[5]),
        ])
        self.ups = nn.ModuleList([
            Up(c[5], c[4], dropout=True),
            Up(c[4] * 2, c[3], dropout=True),
            Up(c[3] * 2, c[2]),
            Up(c[2] * 2, c[1]),
            Up(c[1] * 2, c[0]),
        ])
        self.final = nn.Sequential(nn.ConvTranspose2d(c[0] * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
        for i, u in enumerate(self.ups):
            x = u(x)
            x = torch.cat([x, skips[-2 - i]], dim=1)
        return self.final(x)


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN: sees (dark input, candidate) and outputs a grid of real/fake scores."""

    def __init__(self, in_ch=6, ndf=64):
        super().__init__()

        def block(cin, cout, stride):
            return [nn.Conv2d(cin, cout, 4, stride, 1, bias=False),
                    nn.InstanceNorm2d(cout, affine=True),
                    nn.LeakyReLU(0.2, inplace=False)]

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=False),
            *block(ndf, ndf * 2, 2),
            *block(ndf * 2, ndf * 4, 2),
            *block(ndf * 4, ndf * 8, 1),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, cond, img):
        return self.net(torch.cat([cond, img], dim=1))


def load_generator(path, device="cpu"):
    """Load a generator from `generator_best.pt` (or a full training checkpoint)."""
    ck = torch.load(path, map_location=device)
    ngf = ck.get("ngf", 64)
    g = UNetGenerator(ngf=ngf)
    g.load_state_dict(ck["G"])
    return g.to(device).eval()
