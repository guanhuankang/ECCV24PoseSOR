import os, cv2
import pickle

import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision.ops import masks_to_boxes

def calc_iou(p, t):
    mul = (p*t).sum()
    add = (p+t).sum()
    return mul / (add - mul + 1e-6)

def arange_from_ascending_seq(x):
    """
    Args:
        x: a 1-D tensor starting from 0 to N in an ascending order.
            e.g. 0001122224777 (output=012 01 0123 0 0123)
    Return:
        y: 012010123400123, arange from 0 in each segment
    """
    n = len(x)
    y = torch.zeros_like(x)
    for i in range(1, n):
        if x[i] != x[i-1]:
            continue
        else:
            y[i] = y[i-1] + 1
    return y

def pad1d(x, dim, num, value=0.0):
    """

    Args:
        pad a torch.Tensor along dim (at the end) to be dim=num
        x: any shape torch.Tensor
        dim: int
        repeats: int

    Returns:
        x: where x.shape[dim] = num
    """
    size = list(x.shape)
    size[dim] = num - size[dim]
    assert size[dim] >= 0, "{} < 0".format(size[dim])
    v = torch.ones(size, dtype=x.dtype, device=x.device) * value
    return torch.cat([x, v], dim=dim)

def mask2Boxes(masks):
    """

    Args:
        masks: n, H, W

    Returns:
        bbox: n, 4 [(x1,y1),(x2,y2)] \in [0,1]

    """
    n, H, W = masks.shape
    bbox = masks_to_boxes(masks)
    xi, yi, xa, ya = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    xi = xi / W
    yi = yi / H
    xa = xa / W
    ya = ya / H
    return torch.clamp(torch.stack([xi, yi, xa, ya], dim=1), 0.0, 1.0)

def xyhw2xyxy(bbox):
    """

    Args:
        bbox: N, 4 [0,1]

    Returns:
        bbox: N, 4
    """
    x, y, h, w = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1, y1, x2, y2 = x-w/2., y-h/2., x+w/2., y+h/2.
    return torch.stack([x1, y1, x2, y2], dim=-1)  ## N, 4

def xyxy2xyhw(bbox):
    """

    Args:
        bbox: N, 4 [0,1]

    Returns:
        bbox: N, 4
    """
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x, y, h, w = (x1+x2)/2., (y1+y2)/2., y2-y1, x2-x1
    return torch.stack([x, y, h, w], dim=-1)  ## N, 4
