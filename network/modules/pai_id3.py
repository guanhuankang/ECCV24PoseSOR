
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable

from .registry import ACTOR_INTERACTION_REG
from ..other import Attention, MLPBlock, init_weights_

class CASAFFN(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.cross_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.self_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout_attn)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout3 = nn.Dropout(p=dropout_ffn)
        self.norm3 = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def forward(self, q, qpe, k, kpe, q_mask=None):
        m = None
        if not isinstance(q_mask, type(None)):
            m = (q_mask @ q_mask.transpose(-1, -2)).unsqueeze(1)  ## B, 1, n, n
        q = self.norm1(q + self.dropout1(self.cross_attn(q=q + qpe, k=k + kpe, v=k)))
        q = self.norm2(
            q + self.dropout2(self.self_attn(q=q + qpe, k=q + qpe, v=q, m=m))
        )
        q = self.norm3(q + self.dropout3(self.ffn(q)))
        return q

class PAIHead(nn.Module):
    def __init__(self, embed_dim=256):
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
        self.query_emb = nn.Linear(embed_dim, embed_dim)
        self.heat_emb = nn.Linear(embed_dim, embed_dim)
        self.bbox_head = nn.Linear(embed_dim, 4)
        self.classification_head = nn.Linear(embed_dim, 3)
        init_weights_(self)

    def forward(self, query_emb, joints_emb, mask_features):
        """
        Args:
            query_emb: B, n, C
            joints_emb: B, n, k, C
            mask_features: B, C, H, W
        Return:
            keypoints: B, n, k, 3  (logit)
            heatmaps: B, n, k, H, W (logit)
        """
        B, n, C = query_emb.shape
        mask_features = self.upx4(mask_features)  ## B, C, 4H, 4W
        size = mask_features.shape[2::]  ## (4H, 4W)

        heatmaps = self.heat_emb(joints_emb) @ mask_features.flatten(2).unsqueeze(1)
        heatmaps = heatmaps.unflatten(-1, sizes=size)  ## B, n, k, 4H, 4W

        masks = self.query_emb(query_emb) @ mask_features.flatten(2)
        masks = masks.unflatten(-1, sizes=size)  ## B, n, 4H, 4W
        bboxes = self.bbox_head(query_emb)  ## B, n, 4

        keypoints = self.classification_head(joints_emb)  ## B, n, k, 3

        return {
            "keypoints": keypoints,  ## B, n, k, 3
            "heatmaps": heatmaps,  ## B, n, k, 4H, 4W
            "masks": masks,  ## B, n, 4H, 4W
            "bboxes": bboxes,  ## B, n, 4
        }  ## logit

class AInteraction(nn.Module):
    def __init__(self, num_joints=17, scene_bins=(4, 5), embed_dim=256, num_heads=8, hidden_dim=2048, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.inner_joints_ia = CASAFFN(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout_attn=dropout_attn,
            dropout_ffn=dropout_ffn,
        )
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        init_weights_(self.q_linear)

        self.sub_pose = nn.Sequential(
            nn.Linear(num_joints, int(4 * num_joints)),
            nn.ReLU(),
            nn.LayerNorm(int(4 * num_joints)),
            nn.Linear(int(4 * num_joints), 1),
        )
        self.sub_pose_norm = nn.LayerNorm(embed_dim)
        init_weights_(self.sub_pose_norm)

        self.scene_pool = nn.AdaptiveAvgPool2d(scene_bins)
        self.scene_norm = nn.LayerNorm(embed_dim)
        init_weights_(self.scene_norm)

        self.xi_sa = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.xi_sa_norm = nn.LayerNorm(embed_dim)
        self.xi_sa_drop = nn.Dropout(dropout_attn)
        self.xi_ffn = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.xi_ffn_norm = nn.LayerNorm(embed_dim)
        self.xi_ffn_drop = nn.Dropout(dropout_ffn)

        init_weights_(self.xi_sa)
        init_weights_(self.xi_sa_norm)
        init_weights_(self.xi_sa_drop)
        init_weights_(self.xi_ffn)
        init_weights_(self.xi_ffn_norm)
        init_weights_(self.xi_ffn_drop)

    def forward(self, j, jpe, q, qpe, z, zpe, q_mask):
        """
        Args:
            j, jpe: B, n, k, C
            q, qpe: B, n, C
            z, zpe: B,C,H,W
            q_mask: B, n, 1 (01 value, 0 means visible, 1 means not visible (need masking out))
        Returns:
            q: B, n, C
            j, B, n, k, C
        """
        B, N, K, C = j.shape

        f = z.flatten(2)  ## B, C, HW
        m = (self.q_linear(q) @ f).sigmoid() * 0.0 + 1.0 ## B, n, HW
        f = f.transpose(-1, -2).unsqueeze(1) * m.unsqueeze(-1)  ## B, n, HW, C
        fpe = zpe.flatten(2).transpose(-1, -2).unsqueeze(1).repeat_interleave(N, dim=1)

        # j_mask = torch.repeat_interleave(q_mask, K, dim=1)  ## B, nk, 1
        j_mask = q_mask.flatten(0, 1).unsqueeze(1).repeat_interleave(K+1, dim=1)  ## Bn, K+1, 1
        jq = self.inner_joints_ia(
            q=torch.cat([j.flatten(0, 1), q.flatten(0, 1).unsqueeze(1)], dim=1),  ## Bn, k+1, C
            qpe=torch.cat([jpe.flatten(0, 1), qpe.flatten(0, 1).unsqueeze(1)], dim=1),  ## Bn, k+1, C
            k=f.flatten(0, 1),  ## Bn, HW, C
            kpe=fpe.flatten(0, 1),  ## Bn, HW, C
            q_mask=j_mask,  ## Bn, k+1, c
        ).unflatten(0, sizes=(B, N))  ## B, N, k+1, C
        j, q = jq[:, :, 0:K, :], jq[:, :, -1, :]
        del f, fpe

        sc_tokens = self.scene_norm(self.scene_pool(z).flatten(2).transpose(-1, -2))  ## B, hw, C
        sc_pe = self.scene_pool(zpe).flatten(2).transpose(-1, -2)  ## B, hw, C
        del z, zpe

        sp_tokens = self.sub_pose_norm(q + self.sub_pose(j.transpose(-1, -2)).squeeze(-1))  ## B, n, c
        sp_pe = qpe + self.sub_pose(jpe.transpose(-1, -2)).squeeze(-1)  ## B, n, c

        xq = torch.cat([sp_tokens, sc_tokens], dim=1)  ## B, n+hw, C
        xpe = torch.cat([sp_pe, sc_pe], dim=1)  ## B, n+hw, C
        x_mask = torch.cat([
            q_mask,
            torch.ones((B, sc_tokens.shape[1], 1)).to(q_mask),
        ], 1)  ## B, n+hw, C
        x_mask_square = torch.matmul(x_mask, x_mask.transpose(-1, -2)).unsqueeze(1)  ## B, 1, *, *

        xq = self.xi_sa_norm(xq + self.xi_sa_drop(self.xi_sa(q=xq+xpe, k=xq+xpe, v=xq, m=x_mask_square)))
        xq = self.xi_ffn_norm(xq + self.xi_ffn_drop(self.xi_ffn(xq)))

        sp_tokens = xq[:, 0:N, :]  ## B, n, C
        q = q + sp_tokens  ## B, n, C
        # j = j + self.sp_linear(sp_tokens.unsqueeze(2))  ## B, n, k, C
        return q, j

@ACTOR_INTERACTION_REG.register()
class PAIID3(nn.Module):
    @configurable
    def __init__(
        self,
        num_layers=3,
        num_blocks=3,
        mask_key="res2",
        key_feature="res2",
        embed_dim=256,
        num_heads=8,
        hidden_dim=1024,
        dropout_attn=0.0,
        dropout_ffn=0.0,
        num_joints=17,
        scene_bins=(4, 5),
    ):
        super().__init__()

        self.j = nn.Parameter(torch.zeros((1, num_joints, embed_dim)))
        self.jpe = nn.Parameter(torch.randn((1, num_joints, embed_dim)))

        self.blocks = nn.ModuleList([
            AInteraction(
                num_joints=num_joints,
                # num_layers=num_layers,
                scene_bins=scene_bins,
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout_attn=dropout_attn,
                dropout_ffn=dropout_ffn,
            )
            for _ in range(num_blocks)
        ])
        self.head = PAIHead(embed_dim=embed_dim)
        self.mask_key = mask_key
        self.key_feature = key_feature

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads": cfg.MODEL.COMMON.NUM_HEADS,
            "hidden_dim": cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "dropout_ffn": cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_joints": cfg.MODEL.COMMON.NUM_JOINTS,
            "num_layers": cfg.MODEL.ACTOR_INTERACTION.NUM_LAYERS,
            "num_blocks": cfg.MODEL.ACTOR_INTERACTION.NUM_BLOCKS,
            "mask_key": cfg.MODEL.ACTOR_INTERACTION.MASK_KEY,
            "key_feature": cfg.MODEL.ACTOR_INTERACTION.KEY_FEATURE,
            "scene_bins": cfg.MODEL.ACTOR_INTERACTION.SCENE_BINS,
        }

    def forward(self, q, qpe, feats, feats_pe, q_mask=None):
        """
        Args:
            q, qpe: B, n, C
            feats: dict of B,C,Hi,Wi
            feats_pe: dict of B,C,Hi,Wi
            q_mask: B, n, 1 (01 value, 0 means visible, 1 means not visible (need masking out))
        Returns:
            joints_emb, joints_emb_pe: B, n, k, C
            actor: B, n, C
            out: B, n, k, 3
            aux: list of (B, n, k, 3)
        """
        mask_feat = feats[self.mask_key]
        # mask_feat_pe = feats_pe[self.mask_key]
        z = feats[self.key_feature]
        zpe = feats_pe[self.key_feature]

        j = self.j.expand(len(q), -1, -1).unsqueeze(1) + q.unsqueeze(2)  ## B, n, k, C
        jpe = self.jpe.expand(len(qpe), -1, -1).unsqueeze(1) + qpe.unsqueeze(2)  ## B, n, k, C

        predictions = [self.head(q, j, mask_feat)]
        for block in self.blocks:
            q, j = block(j=j, jpe=jpe, q=q, qpe=qpe, z=z, zpe=zpe, q_mask=q_mask)
            predictions.append(self.head(q, j, mask_feat))

        out = predictions[-1]
        aux = predictions[0:-1]

        return j, jpe, q, qpe, out, aux
