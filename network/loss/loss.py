import torch
import torch.nn.functional as F
import torchvision
from ..utils import xyxy2xyhw

def batch_mask_loss(preds, targets, cfg=None):
    """
    CE loss + dice loss

    Args:
        preds: B,* logits
        targets: B,* binary
    Returns:
        loss: B,1
    """
    if cfg is None:
        ce_loss_weight = 1.0
        dice_loss_weight = 1.0
    else:
        ce_loss_weight = cfg.LOSS.MASK_CE_COST
        dice_loss_weight = cfg.LOSS.MASK_DICE_COST
        
    preds = preds.flatten(1)  ## B,-1
    targets = targets.flatten(1)  ## B, -1
    sig_preds = torch.sigmoid(preds)  ## B, -1

    ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none").mean(dim=-1)  ## B
    dice_loss = 1.0 - ((2. * sig_preds * targets).sum(dim=-1)+1.0) / ((sig_preds+targets).sum(dim=-1) + 1.0)  ## B
    return ce_loss * ce_loss_weight + dice_loss * dice_loss_weight


def batch_mask_loss_in_points(preds, targets, cfg=None):
    """
    preds: *, H, W
    targets: *, H, W
    """
    H, W = preds.shape[-2::]
    K = cfg.LOSS.NUM_POINTS
    if H*W <= K:
        return batch_mask_loss(preds=preds, targets=targets, cfg=cfg)
    
    assert targets.shape[-2::]==preds.shape[-2::]
    khi = torch.randint(low=0, high=H, size=(K,)).to(preds.device).long().reshape(-1)
    kwi = torch.randint(low=0, high=W, size=(K,)).to(preds.device).long().reshape(-1)
    return batch_mask_loss(
        preds=preds.reshape(-1, H, W)[:, khi, kwi],
        targets=targets.reshape(-1, H, W)[:, khi, kwi],
        cfg=cfg
    )


# def batch_cls_loss(preds, targets):
#     return F.binary_cross_entropy_with_logits(preds, targets, reduction="none").flatten(1).mean(dim=-1)

def batch_bbox_loss(box1, box2, cfg=None):
    """
    boxes in [(x1,y1),(x2,y2)]
    Args:
        box1: N, 4 [0, 1]
        box2: N, 4 [0, 1]

    Returns:
        loss: N
    """
    if cfg is None:
        bbox_l1_weight = 1.0
        bbox_giou_weight = 1.0
    else:
        bbox_l1_weight = cfg.LOSS.BBOX_L1_COST
        bbox_giou_weight = cfg.LOSS.BBOX_GIOU_COST

    version = [int(_) for _ in torchvision.__version__.split("+")[0].split(".")]
    if version[1] >= 15:
        gloss = torchvision.ops.generalized_box_iou_loss(box1, box2)  ## N
    else:
        gloss = -torch.diag(torchvision.ops.generalized_box_iou(box1, box2))  ## N
    l1loss = F.l1_loss( xyxy2xyhw(box1), xyxy2xyhw(box2), reduction="none").mean(dim=-1)
    return l1loss * bbox_l1_weight + gloss * bbox_giou_weight


def gaussian_focal_loss(preds, targets):
    """ ref: https://arxiv.org/pdf/1808.01244.pdf """
    alpha = 2.0
    beta = 4.0
    high_confidence = 0.99
    eps = torch.zeros_like(targets) + 1e-6
    return -torch.where(
        targets >= high_confidence,
        torch.pow(1.0 - preds, alpha) * torch.log(torch.maximum(preds, eps)),
        torch.pow(1.0 - targets, beta) * torch.pow(preds, alpha) * torch.log(torch.maximum(1.0 - preds, eps))
    )  ## *

def heatmap_loss_in_points(heatmaps, coords, cfg=None):
    """
    Args:
        heatmaps: (*, H, W) [logits]
        coords: (*, 2or3) \in [0.0, 1.0]
    Return:
        heatmap_loss: loss (scalar)
    """
    H, W = heatmaps.shape[-2::]
    size = heatmaps.shape[0:-2]
    D = coords.shape[-1]
    heatmaps = heatmaps.reshape(-1, H, W)  ## n, H, W
    coords = coords.reshape(-1, D)  ## n, D
    assert len(heatmaps) == len(coords)

    # K = cfg.LOSS.NUM_POINTS
    sigma = cfg.LOSS.SIGMA

    ## get GT
    x0 = torch.ones_like(heatmaps) * coords[:, 0][:, None, None]  ## n, H, W
    y0 = torch.ones_like(heatmaps) * coords[:, 1][:, None, None]  ## n, H, W
    x = torch.linspace(0.0, 1.0, W).repeat(H, 1).unsqueeze(0).to(heatmaps)  ## 1, H, W
    y = torch.linspace(0.0, 1.0, H)[:, None].repeat(1, W).unsqueeze(0).to(heatmaps)  ## 1, H, W
    xy = torch.stack([x, y], dim=-1)  ## 1, H, W, 2
    u = torch.stack([x0, y0], dim=-1)  ## n, H, W, 2
    z = torch.exp(-torch.sum((xy-u)*(xy-u), dim=-1) / (2.0 * sigma * sigma))  ## n, H, W
    max_z = z.flatten(1).max(dim=-1)[0][:, None, None]  ## n, 1, 1
    z = z / (max_z+1e-6)  ## n, H, W
    # assert z.min() >= 0.0 and z.max() <= 1.0

    # return gaussian_focal_loss(heatmaps.sigmoid(), z).unflatten(0, sizes=size).flatten(-2).mean(dim=-1)
    mask = z.gt(0.1).float()  ## ~= 500 points, n, H, W
    K = 100
    khi = torch.randint(low=0, high=H, size=(K,)).to(heatmaps.device).long().reshape(-1)
    kwi = torch.randint(low=0, high=W, size=(K,)).to(heatmaps.device).long().reshape(-1)
    mask[:, khi, kwi] = 1.0  ## n, H, W
    bi, hi, wi = torch.where(mask.gt(0.5))

    hm_size = heatmaps.shape[-2::]
    heatmaps = torch.softmax(heatmaps.flatten(-2), dim=-1).unflatten(-1, hm_size)
    return gaussian_focal_loss(
        preds=heatmaps[bi, hi, wi],
        targets=z[bi, hi, wi],
    ).mean()
