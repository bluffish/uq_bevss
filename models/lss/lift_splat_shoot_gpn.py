from models.lss.lift_splat_shoot import *
from models.gpn.density import Density, Evidence


class BevEncodeGPN(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncodeGPN, self).__init__()

        self.outC = outC

        trunk = resnet18(zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)

        self.latent_size = 16

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, self.latent_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.latent_size),
            nn.ReLU(inplace=True),
        )

        self.flow = Density(dim_latent=self.latent_size, num_mixture_elements=outC)
        self.evidence = Evidence(scale='latent-new')

        self.last = nn.Conv2d(outC, outC, kernel_size=3, padding=1)

        self.p_c = None
        self.tsne = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        x_b = torch.clone(x)

        x = x.permute(0, 2, 3, 1).to(x.device)
        x = x.reshape(-1, self.latent_size)

        self.p_c = self.p_c.to(x.device)

        log_q_ft_per_class = self.flow(x) + self.p_c.view(1, -1).log()

        beta = self.evidence(
            log_q_ft_per_class, dim=self.latent_size,
            further_scale=2.0).exp()

        beta = beta.reshape(-1, 200, 200, self.outC).permute(0, 3, 1, 2).contiguous()

        if self.last is not None:
            beta = self.last(beta.log()).exp()
        alpha = beta + 1

        if self.tsne:
            return x_b
        else:
            return alpha


class LiftSplatShootGPN(LiftSplatShoot):
    def __init__(self, outC=4, use_seg=False):
        super(LiftSplatShootGPN, self).__init__(outC=outC, use_seg=use_seg)

        self.bevencode = BevEncodeGPN(inC=self.camC, outC=self.outC)
        self.use_seg = use_seg

    def forward(self, x, rots, trans, intrins, extrins, post_rots, post_trans):
        p, s = super().forward(x, rots, trans, intrins, extrins, post_rots, post_trans)

        if s is not None:
            return p, s.relu() + 1
        else:
            return p, None
