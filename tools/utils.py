import numpy as np
import torch.nn
from sklearn.metrics import *

from datasets.nuscenes import *
from models.cvt.cross_view_transformer import *
from models.lss.lift_splat_shoot import LiftSplatShoot, LiftSplatShootENN
from models.lss.lift_splat_shoot_ensemble import LiftSplatShootEnsemble
from models.lss.lift_splat_shoot_gpn import LiftSplatShootGPN
from models.lss.lift_splat_shoot_dropout import LiftSplatShootDropout

from tools.loss import *
from tools.uncertainty import *

colors = torch.tensor([
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
])


def patch_metrics(uncertainty_scores, uncertainty_labels, sample_size=1_000_000):
    thresholds = np.linspace(0, 1, 10)
    pavpus = []
    agcs = []
    ugis = []

    for threshold in thresholds:
        pavpu, agc, ugi = calculate_pavpu(uncertainty_scores, uncertainty_labels, uncertainty_threshold=threshold)
        pavpus.append(pavpu)
        agcs.append(agc)
        ugis.append(ugi)

    return pavpus, agcs, ugis, thresholds, auc(thresholds, pavpus), auc(thresholds, agcs), auc(thresholds, ugis)


def calculate_pavpu(uncertainty_scores, uncertainty_labels, accuracy_threshold=0.5, uncertainty_threshold=0.2, window_size=4):
    ac, ic, au, iu = 0., 0., 0., 0.

    anchor = (0, 0)
    last_anchor = (uncertainty_labels.shape[1] - window_size, uncertainty_labels.shape[2] - window_size)

    while anchor != last_anchor:
        label_window = uncertainty_labels[:, anchor[0]:anchor[0] + window_size, anchor[1]:anchor[1] + window_size]
        uncertainty_window = uncertainty_scores[:, anchor[0]:anchor[0] + window_size, anchor[1]:anchor[1] + window_size]

        accuracy = torch.sum(label_window, dim=(1, 2)) / (window_size ** 2)
        avg_uncertainty = torch.mean(uncertainty_window, dim=(1, 2))

        accurate = accuracy < accuracy_threshold
        uncertain = avg_uncertainty >= uncertainty_threshold

        au += torch.sum(accurate & uncertain)
        ac += torch.sum(accurate & ~uncertain)
        iu += torch.sum(~accurate & uncertain)
        ic += torch.sum(~accurate & ~uncertain)

        if anchor[1] < uncertainty_labels.shape[1] - window_size:
            anchor = (anchor[0], anchor[1] + window_size)
        else:
            anchor = (anchor[0] + window_size, 0)

    a_given_c = ac / (ac + ic + 1e-10)
    u_given_i = iu / (ic + iu + 1e-10)

    pavpu = (ac + iu) / (ac + au + ic + iu + 1e-10)

    return pavpu.item(), a_given_c.item(), u_given_i.item()


def roc_pr(uncertainty_scores, uncertainty_labels, sample_size=1_000_000):
    y_true = uncertainty_labels.flatten()
    y_score = uncertainty_scores.flatten()

    indices = np.random.choice(y_true.shape[0], sample_size, replace=False)

    y_true = y_true[indices]
    y_score = y_score[indices]

    pr, rec, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    aupr = auc(rec, pr)
    auroc = auc(fpr, tpr)

    no_skill = torch.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, aupr, no_skill


def parse(imgs, gt):
    seg = imgs[:, :, 3:, :, :].view(-1, 3, 128, 352)
    back = ~(seg[:, 0, :, :] + seg[:, 1, :, :] + seg[:, 2, :, :]).bool()[:, None, :, :]
    seg = torch.cat((seg, back), dim=1)
    i = imgs[:, :, :3, :, :]

    if gt:
        back = ~(imgs[:, :, 3, :, :] + imgs[:, :, 4, :, :] + imgs[:, :, 5, :, :]).bool()[:, :, None, :, :]

        return torch.cat((imgs, back), dim=2), seg

    return i, seg


def get_iou(preds, labels):
    classes = preds.shape[1]
    intersect = [0]*classes
    union = [0]*classes

    with torch.no_grad():
        for i in range(classes):
            pred = (preds[:, i, :, :] >= .5)
            tgt = labels[:, i, :, :].bool()
            intersect[i] = (pred & tgt).sum().float().item()
            union[i] = (pred | tgt).sum().float().item()

    return intersect, union


backbones = {
    'lss': [LiftSplatShoot, LiftSplatShootENN, LiftSplatShootGPN, LiftSplatShootDropout, LiftSplatShootEnsemble],
    'cvt': [CrossViewTransformer, CrossViewTransformerENN, CrossViewTransformerGPN, CrossViewTransformerDropout,
            CrossViewTransformerEnsemble]
}


def get_model(type, backbone, num_classes, device, use_seg=False):
    weights = torch.tensor([3.0, 1.0, 2.0, 1.0]).to(device)

    if type == 'baseline':
        activation = softmax
        loss_fn = CELoss(weights=weights.cuda(device)).cuda(device)
        model = backbones[backbone][0](outC=num_classes, use_seg=use_seg)
    elif type == 'enn':
        activation = activate_uce
        loss_fn = UCELoss(weights=weights).cuda(device)
        model = backbones[backbone][1](outC=num_classes, use_seg=use_seg)
    elif type == 'postnet':
        activation = activate_uce
        loss_fn = UCELoss(weights=weights).cuda(device)
        model = backbones[backbone][2](outC=num_classes, use_seg=use_seg)
    elif type == 'dropout':
        activation = softmax
        loss_fn = CELoss(weights=weights.cuda(device)).cuda(device)
        model = backbones[backbone][3](outC=num_classes, use_seg=use_seg)
    elif type == 'ensemble':
        activation = softmax
        loss_fn = CELoss(weights=weights.cuda(device)).cuda(device)
        model = backbones[backbone][4](outC=num_classes, use_seg=use_seg)
    else:
        raise ValueError("Please pick a valid model type.")

    return activation, loss_fn, model


def map(img, m=False):
    if not m:
        dense = img.detach().cpu().numpy().argmax(-1)
    else:
        dense = img.detach().cpu().numpy()

    rgb = np.zeros((*dense.shape, 3))
    for label, color in enumerate(colors):
        rgb[dense == label] = color
    return rgb


def save_pred(pred, labels, out_path):
    pred = map(pred[0].permute(1, 2, 0))
    labels = map(labels[0].permute(1, 2, 0))

    cv2.imwrite(os.path.join(out_path, "pred.jpg"), pred)
    cv2.imwrite(os.path.join(out_path, "label.jpg"), labels)

    return pred, labels
