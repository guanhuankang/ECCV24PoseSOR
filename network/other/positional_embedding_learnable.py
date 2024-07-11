import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbeddingLearnable(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.ape = nn.parameter.Parameter(torch.zeros((embed_dim, 25, 25)), requires_grad=True)
        nn.init.trunc_normal_(self.ape)

    def forward(self, x):
        """
        x: B, C, H, W
        return: B, C, H, W
        """
        ape = F.interpolate(self.ape.unsqueeze(0), size=x.shape[2::], mode="bilinear")  ## 1, C, H, W
        return ape.expand(len(x), -1, -1, -1)  ## B, C, H, W
