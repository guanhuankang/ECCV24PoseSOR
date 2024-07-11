import os
import copy
import torch
import albumentations as A
import numpy as np
from .utils import read_image, parse_anno, merge_masks


def sor_dataset_mapper_train(image_dict, cfg):
    image_dict = copy.deepcopy(image_dict)
    H, W = image_dict["height"], image_dict["width"]
    image = read_image(image_dict["file_name"], format="RGB", cfg=cfg)
    masks = [parse_anno(anno, H, W) for anno in image_dict["annotations"]]
    sal_map = merge_masks(masks, H, W)

    ## data aug
    additional_targets = dict(("mask_{}".format(i), "mask") for i in range(len(masks)))
    additional_targets["sal_map"] = "mask"
    transform = A.Compose([
        # A.HorizontalFlip(p=0.5),
        A.Resize(cfg.INPUT.TRAIN_IMAGE_SIZE, cfg.INPUT.TRAIN_IMAGE_SIZE),
        # A.ColorJitter(),
    ],
        additional_targets=additional_targets
    )
    aug = transform(image=image, sal_map=sal_map, **dict(("mask_{}".format(i), masks[i]) for i in range(len(masks))))
    image = aug["image"]
    masks = [aug["mask_{}".format(i)] for i in range(len(masks))]
    sal_map = aug["sal_map"]

    ## toTensor
    image = torch.from_numpy(image).permute(2, 0, 1).float()  ## 3, H, W
    masks = torch.stack([torch.from_numpy(m).float() for m in masks], dim=0)  ## K, H, W

    category_ids = torch.tensor([anno["category_id"] for anno in image_dict["annotations"]], dtype=torch.long)  ## K
    # is_persons = torch.tensor([anno["class_id"]==1 for anno in image_dict["annotations"]]).float()  ## K
    keypoints = torch.stack([
        torch.tensor(anno["keypoints"]).float().reshape(-1, 3) / torch.tensor([[W, H, 1]]).float()
        for anno in image_dict["annotations"]
    ], dim=0)  ## K, 17, 3
    x1y1wh2xyxy = lambda b: [b[0], b[1], b[0]+b[2], b[1]+b[3]]
    bboxes = torch.tensor([x1y1wh2xyxy(anno["bbox"]) for anno in image_dict["annotations"]])
    bboxes = torch.clamp(bboxes / torch.tensor([[W, H, W, H]]).float(), 0.0, 1.0)  ## K, 4: x1,y1,x2,y2

    background = 1.0 - torch.tensor(sal_map)
    next_targets = [masks[i] for i in torch.argsort(category_ids) if category_ids[i] > 0][1::] + [background]
    next_targets = torch.stack(next_targets, dim=0)  ## K, H, W

    if np.random.rand() < 0.5:
        ## random flip
        image = torch.flip(image, dims=[-1])
        masks = torch.flip(masks, dims=[-1])
        next_targets = torch.flip(next_targets, dims=[-1])
        keypoints[:, :, 0] = 1.0 - keypoints[:, :, 0]
        bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]

    return {
        "image": image,  ## 3, H, W
        "masks": masks,  ## K, H, W
        "keypoints": keypoints,  ## K, 17, 3
        "bboxes": bboxes,  ## K, 4: x1,y1,x2,y2
        "category_ids": category_ids,  ## K
        "next_targets": next_targets,  ## K, H, W
        "height": image_dict["height"],
        "width": image_dict["width"],
        "image_name": os.path.basename(image_dict["file_name"]).split(".")[0],
        # "image_name": str(image_dict["id"]),
        # "is_persons": is_persons,  ## K
    }