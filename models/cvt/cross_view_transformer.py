from models.cvt.decoder import *
from models.cvt.encoder import *
from models.gpn.density import Density, Evidence
from models.seg.unet import UNet

import cv2


def convert(intrins):
    intrins[:, :, 0, 0] *= W / 1600
    intrins[:, :, 0, 2] *= W / 1600
    intrins[:, :, 1, 1] *= (H + O) / 900
    intrins[:, :, 1, 2] *= (H + O) / 900
    intrins[:, :, 1, 2] -= O
    return intrins


class Shrink(nn.Module):
    def __init__(self):
        super(Shrink, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(7, 5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        dim_last: int = 64,
        outC: int = 4,
        use_seg=False,
    ):
        super().__init__()
        print("Initializing CVT model")

        self.encoder = Encoder(use_seg=use_seg)
        self.decoder = Decoder(128, [128, 128, 64])

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, outC, 1))

        if use_seg:
            self.seg = UNet(n_channels=3, n_classes=outC)

        self.use_seg = use_seg
        self.drop = False

    def forward(self, imgs, rots, trans, intrins, extrins, post_rots, post_trans):
        batch = {
            'image': imgs,
            'intrinsics': convert(intrins),
            'extrinsics': extrins
        }

        if self.use_seg and imgs.shape[2] == 3:
            seg = self.seg(imgs.view(-1, 3, 128, 352))
            batch['image'] = torch.cat((imgs, seg.view(-1, 6, 4, 128, 352)), dim=2)
        else: seg = None

        x, atts = self.encoder(batch)
        y = self.decoder(x)

        if self.drop:
            self.train()

        z = self.to_logits(y)

        if self.drop:
            self.eval()

        return z, seg


class CrossViewTransformerENN(CrossViewTransformer):
    def __init__(
        self,
        dim_last: int = 64,
        outC: int = 4,
        use_seg=False
    ):
        super(CrossViewTransformerENN, self).__init__(outC=outC, dim_last=dim_last, use_seg=use_seg)

    def forward(self, imgs, rots, trans, intrins, extrins, post_rots, post_trans, return_att=False):
        beta, beta_s = super().forward(imgs, rots, trans, intrins, extrins, post_rots, post_trans)

        beta = beta.relu()

        if beta_s is not None:
            alpha_s = beta_s.relu() + 1
        else:
            alpha_s = None

        alpha = beta + 1

        return alpha, alpha_s


class CrossViewTransformerDropout(CrossViewTransformer):
    def __init__(
        self,
        dim_last: int = 64,
        outC: int = 4,
        use_seg=False,
    ):
        super(CrossViewTransformerDropout, self).__init__(outC=outC, dim_last=dim_last, use_seg=use_seg)

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Dropout2d(.3),
            nn.Conv2d(dim_last, outC, 1)
        )

        self.use_seg = use_seg
        self.tests = 1

    def forward(self, imgs, rots, trans, intrins, extrins, post_rots, post_trans):
        outputs = []
        seg_outputs = []

        for i in range(self.tests):
            p, s = super().forward(imgs.clone(), rots, trans, intrins.clone(), extrins.clone(), post_rots, post_trans)
            outputs.append(p)
            seg_outputs.append(s)

        outputs = torch.stack(outputs)
        seg_outputs = torch.stack(seg_outputs) if seg_outputs[0] is not None else None

        return outputs, seg_outputs


class CrossViewTransformerEnsemble(nn.Module):
    def __init__(
        self,
        dim_last: int = 64,
        outC: int = 4,
        use_seg=False,
    ):
        super(CrossViewTransformerEnsemble, self).__init__()

        num_models = 5
        self.models = nn.ModuleList([CrossViewTransformer(outC=outC, use_seg=use_seg) for _ in range(num_models)])

    def forward(self, imgs, rots, trans, intrins, extrins, post_rots, post_trans):
        outputs = []
        seg_outputs = []

        for model in self.models:
            p, s = model(imgs.clone(), rots, trans, intrins.clone(), extrins.clone(), post_rots, post_trans)
            outputs.append(p)
            seg_outputs.append(s)

        outputs = torch.stack(outputs)
        seg_outputs = torch.stack(seg_outputs) if seg_outputs[0] is not None else None

        return outputs, seg_outputs

class CrossViewTransformerGPN(CrossViewTransformer):
    def __init__(
            self,
            dim_last: int = 64,
            outC: int = 4,
            use_seg=False
    ):
        super(CrossViewTransformerGPN, self).__init__(outC=outC, dim_last=dim_last, use_seg=use_seg)

        self.outC = outC
        self.latent_size = 16
        self.flow = Density(dim_latent=self.latent_size, num_mixture_elements=outC)
        self.evidence = Evidence(scale='latent-new')

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, self.latent_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.latent_size),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Conv2d(outC, outC, 1)
        self.p_c = None

    def forward(self, imgs, rots, trans, intrins, extrins, post_rots, post_trans):
        batch = {
            'image': imgs,
            'intrinsics': convert(intrins),
            'extrinsics': extrins
        }

        if self.use_seg and imgs.shape[2] == 3:
            seg = self.seg(imgs.view(-1, 3, 128, 352))
            batch['image'] = torch.cat((imgs, seg.view(-1, 6, 4, 128, 352)), dim=2)
        else: seg = None

        x, atts = self.encoder(batch)
        x = self.decoder(x)
        x = self.to_logits(x)

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

        if seg is None:
            return alpha.clamp(min=1e-4), seg
        else:
            return alpha.clamp(min=1e-4), seg.relu() + 1