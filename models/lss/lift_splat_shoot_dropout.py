from models.lss.lift_splat_shoot import *

import cv2
import matplotlib.pyplot as plt

class LiftSplatShootDropout(LiftSplatShoot):
    def __init__(self, outC=4, use_seg=False):
        super(LiftSplatShootDropout, self).__init__(outC=outC,
                                                    use_seg=use_seg)

        self.bevencode.up1.conv = nn.Sequential(
            nn.Conv2d(self.bevencode.up1.in_channels, self.bevencode.up1.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.bevencode.up1.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bevencode.up1.out_channels, self.bevencode.up1.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.bevencode.up1.out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(.3),
        )

        self.bevencode.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(.3),
            nn.Conv2d(128, self.outC, kernel_size=1, padding=0),
        )

        self.tests = 1
        self.use_seg = use_seg

    def forward(self, x, rots, trans, intrins, extrins, post_rots, post_trans):
        outputs = []
        seg_outputs = []

        for i in range(self.tests):
            p, s = super().forward(x, rots, trans, intrins, extrins, post_rots, post_trans)

            outputs.append(p)
            seg_outputs.append(s)

        outputs = torch.stack(outputs)
        seg_outputs = torch.stack(seg_outputs) if seg_outputs[0] is not None else None

        return outputs, seg_outputs

