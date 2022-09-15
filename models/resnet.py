from typing import Optional, List
import torch.nn as nn
import pytorch_lightning as pl


class ResNt(pl.LightningModule):
    def __init__(self, *,
                 d_numerical: int, categories: Optional[List[int]], d_embedding: int,
                 d: int, d_hidden_factor: float, n_layers: int,
                 activation: str, normalization: str,
                 hidden_dropout: float, residual_dropout: float, d_out: int) -> None:
        super(ResNt, self).__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)
