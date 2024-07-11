import torch

def compute_mask_loss(pred_masks, gt_masks):
    """
    Args:
        pred_masks: B, nq, h, w
        gt_masks: list of (K_i, H, W)
    Return:
        pass
    """