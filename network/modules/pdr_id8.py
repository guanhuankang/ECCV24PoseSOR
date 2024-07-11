import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable

from .registry import POSE_SHIFT_REG
from ..other import Attention, MLPBlock, init_weights_

class PDRHead(nn.Module):
    def __init__(self, embed_dim=256, max_rank_level=20):
        super().__init__()
        self.upx4 = nn.Sequential(
            # nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(embed_dim),
            # nn.ReLU(),
            # nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(embed_dim),
            # nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1)
        )
        self.nxt_emb = nn.Linear(embed_dim, embed_dim)
        self.rank_head = nn.Linear(embed_dim, max_rank_level)
        init_weights_(self)

    def forward(self, q, nxt, mask_features):
        """
        Args:
            q, nxt: B, n, C
            mask_features: B, C, H, W
        Return:
            ranks: B, n, rank_max
            partitions: B, n, 4H, 4W
        """
        mask_features = self.upx4(mask_features)  ## B, C, 4H, 4W
        size = mask_features.shape[2::]  ## (4H, 4W)

        ranks = self.rank_head(q)  ## B, n, rank_max_level
        partitions = self.nxt_emb(nxt) @ mask_features.flatten(2)
        partitions = partitions.unflatten(-1, sizes=size)  ## B, n, 4H, 4W

        return {
            "partitions": partitions,  ## B, n, 4H, 4W
            "ranks": ranks,  ## B, n, max_rank_level
        }  ## logit

@POSE_SHIFT_REG.register()
class PDRID8(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, mask_key="res2", num_joints=17, max_rank_level=20):
        super().__init__()
        # self.directional_conv = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim, 1),
        #     nn.BatchNorm2d(embed_dim),
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=(num_joints, 1)),
        #     nn.BatchNorm2d(embed_dim)
        # )
        #
        # self.mlpMixerC = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.LayerNorm(embed_dim),
        #     nn.ReLU()
        # )
        # self.mlpMixerK = nn.Sequential(
        #     nn.Linear(num_joints, 2),
        #     nn.LayerNorm(2),
        #     nn.ReLU()
        # )
        # self.mlpMixerC2 = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.LayerNorm(embed_dim),
        #     nn.ReLU()
        # )

        self.head = PDRHead(embed_dim=embed_dim, max_rank_level=max_rank_level)
        self.mask_key = mask_key

        init_weights_(self)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            # "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            # "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            # "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            # "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_joints":  cfg.MODEL.COMMON.NUM_JOINTS,
            "max_rank_level": cfg.MODEL.COMMON.MAX_RANK_LEVEL,
            "mask_key":       cfg.MODEL.POSE_SHIFT.MASK_KEY,
        }

    def forward(self, q, qpe, pose, pose_pe, feats, feats_pe, q_mask=None):
        """
        Args:
            q, qpe: B, n, C
            pose, pose_pe: B, n, k, C
            feats, feats_pe: dict of features
            q_mask: B, n, 1
        Return:
            q_pose: B, n, C
            out/aux: {"partition": B, n, H, W}
        """
        mask_features = feats[self.mask_key]
        out = self.head(q=q, nxt=q, mask_features=mask_features)

        return out, []
