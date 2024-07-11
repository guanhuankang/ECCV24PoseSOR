import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable

from .registry import POSE_SHIFT_REG
from ..other import Attention, MLPBlock, init_weights_


class MLPMixer(nn.Module):
    def __init__(self, num_nodes=8, hidden_nodes=32, out_nodes=None, embed_dim=256, hidden_dim=2048, out_dims=None):
        super().__init__()
        out_nodes = num_nodes if isinstance(out_nodes, type(None)) else out_nodes
        out_dims = embed_dim if isinstance(out_dims, type(None)) else out_dims
        self.mlp1 = nn.Sequential(
            nn.Linear(num_nodes, hidden_nodes),
            nn.ReLU(),
            nn.LayerNorm(hidden_nodes),
            nn.Linear(hidden_nodes, out_nodes)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dims)
        )

        init_weights_(self.mlp1)
        init_weights_(self.mlp2)

    def forward(self, x):
        """" x: *.num_nodes, embed_dim """
        x = self.mlp1(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.mlp2(x)
        return x

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
            "q": q,
            "nxt": nxt,
        }  ## logit

@POSE_SHIFT_REG.register()
class PDR(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, mask_key="res2", num_joints=17, num_directions=8, max_rank_level=20,
                 num_heads=8, hidden_dim=2048, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.to_directions = nn.Sequential(
            nn.Linear(num_joints, int(4 * num_joints)),
            nn.ReLU(),
            nn.LayerNorm(int(4 * num_joints)),
            nn.Linear(int(4 * num_joints), num_directions),
        )
        init_weights_(self.to_directions)

        self.directional_cues = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        init_weights_(self.directional_cues)

        self.coordinate_cues = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        init_weights_(self.coordinate_cues)

        self.bg_coord_rep = nn.Parameter(torch.rand(1, num_joints, embed_dim))
        self.bg_linear = nn.Linear(embed_dim, embed_dim)
        self.bg_pool = nn.AdaptiveAvgPool2d(1)
        init_weights_(self.bg_linear)

        self.d_mmixer = MLPMixer(
            num_nodes=num_directions,
            hidden_nodes=int(4 * num_directions),
            out_nodes=1,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dims=embed_dim,
        )

        self.share_mlp = nn.Sequential(
            nn.Linear(embed_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim)
        )
        init_weights_(self.share_mlp)

        self.share_linear = nn.Linear(embed_dim + embed_dim, embed_dim)
        self.agg_head = nn.Linear(embed_dim + embed_dim + embed_dim, embed_dim)
        self.nxt_head = nn.Linear(embed_dim + embed_dim + embed_dim, embed_dim)
        init_weights_(self.share_linear)
        init_weights_(self.agg_head)
        init_weights_(self.nxt_head)

        self.head = PDRHead(embed_dim=embed_dim, max_rank_level=max_rank_level)
        self.mask_key = mask_key

        init_weights_(self)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_joints":  cfg.MODEL.COMMON.NUM_JOINTS,
            "max_rank_level": cfg.MODEL.COMMON.MAX_RANK_LEVEL,
            "mask_key":       cfg.MODEL.POSE_SHIFT.MASK_KEY,
            "num_directions": cfg.MODEL.POSE_SHIFT.NUM_DIRECTIONS,
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
        z = feats[self.mask_key]  ## B, C, H, W
        # zpe = feats_pe[self.mask_key]  ## B, C, H, W

        dc = self.directional_cues(self.to_directions(pose.transpose(-1, -2)).transpose(-1, -2))  ## B, n, d, C
        cc = self.coordinate_cues(pose)  ## B, n, k, C
        cc = torch.cat([cc, self.bg_coord_rep.unsqueeze(0).expand(len(q), -1, -1, -1)], dim=1)  ## B, n+1, k, C
        bg = self.bg_linear(self.bg_pool(z).squeeze(-1).transpose(-1, -2) - torch.mean(q, dim=1, keepdim=True))  ## B, 1, C

        d2c = torch.matmul(dc, cc.flatten(1, 2).unsqueeze(1).transpose(-1, -2))  ## B, n, d, (n+1)k
        qc = torch.cat([q, bg], dim=1).unsqueeze(2) + cc  ## B, n+1, k, C
        dt = torch.softmax(d2c, dim=-1) @ qc.flatten(1, 2).unsqueeze(1)  ## B, n, d, C

        n = dt.shape[1]
        nxt_repr = self.d_mmixer(dt).squeeze(2)  ## B, n, C !!!
        sha_repr = self.share_linear(torch.cat([nxt_repr, q], dim=-1))  ## B, n, C
        mat_repr = self.share_mlp(torch.cat([
            sha_repr.unsqueeze(2).repeat_interleave(n, dim=2),
            sha_repr.unsqueeze(1).repeat_interleave(n, dim=1),
        ], dim=-1))  ## B, n, n, C
        sha_repr = torch.mean(mat_repr, dim=2)  ## B, n, C !!!
        q = self.agg_head(torch.cat([q, nxt_repr, sha_repr], dim=-1))
        nxt = self.nxt_head(torch.cat([q, nxt_repr, sha_repr], dim=-1))

        out = self.head(q=q, nxt=nxt, mask_features=z)

        return out, []
