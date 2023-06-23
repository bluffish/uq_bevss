from models.lss.lift_splat_shoot import *


class LiftSplatShootEnsemble(nn.Module):
    def __init__(self, outC=4, use_seg=False):
        super(LiftSplatShootEnsemble, self).__init__()

        num_models = 5
        self.models = nn.ModuleList([LiftSplatShoot(outC=outC, use_seg=use_seg) for _ in range(num_models)])
        self.use_seg = use_seg

    def forward(self, x, rots, trans, intrins, extrins, post_rots, post_trans):
        outputs = []
        seg_outputs = []

        for model in self.models:
            p, s = model(x, rots, trans, intrins, extrins, post_rots, post_trans)
            outputs.append(p)
            seg_outputs.append(s)

        outputs = torch.stack(outputs)
        seg_outputs = torch.stack(seg_outputs) if seg_outputs[0] is not None else None

        return outputs, seg_outputs