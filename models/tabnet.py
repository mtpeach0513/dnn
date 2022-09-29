from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl

import models.sparsemax as sparsemax
from utils.configure import Config


class TabNetModel(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim=1,
                 n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02,
                 mask_type='sparsemax'):
        super(TabNetModel, self).__init__()

        self.tabnet = TabNet(
            input_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            cat_idxs,
            cat_dims,
            cat_emb_dim,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            mask_type
        )
        self.save_hyperparameters()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x, _ = self.tabnet(x)
        x = x.squeeze(-1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=Config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, min_lr=0, verbose=True
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idex, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        return y_hat


def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain_value)


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain_value)


# Ghost Batch Normalization
class GBN(nn.Module):
    def __init__(self, input_dim: int,
                 virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x: Tensor):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


class TabNetEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim=1,
                 n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02, mask_type='sparsemax'):
        super(TabNetEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multitask = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        if self.n_shared > 0:
            shared_feature_transform = nn.ModuleList(
                [
                    nn.Linear(self.input_dim, 2 * (n_d + n_a), bias=False) if i == 0
                    else nn.Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    for i in range(self.n_shared)
                ]
            )
        else:
            shared_feature_transform = None

        self.initial_splitter = FeatureTransformer(
            self.input_dim,
            n_d + n_a,
            shared_feature_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum
        )

        self.feature_transformers = nn.ModuleList(
            [
                FeatureTransformer(
                    self.input_dim, n_d + n_a, shared_feature_transform,
                    n_glu_independent=self.n_independent,
                    virtual_batch_size=self.virtual_batch_size,
                    momentum=momentum
                )
                for _ in range(n_steps)
            ]
        )

        self.attn_transformers = nn.ModuleList(
            [
                AttentiveTransformer(
                    n_a, self.input_dim,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum, mask_type=self.mask_type
                )
                for _ in range(n_steps)
            ]
        )

    def forward(self, x, prior=None):
        x = self.initial_bn(x)

        if prior is None:
            prior = torch.ones(x.shape)

        M_loss = 0
        attn = self.initial_splitter(x)[:, self.n_d:]

        steps_output = []
        for step in range(self.n_steps):
            M = self.attn_transformers[step](prior, attn)
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feature_transformers[step](masked_x)
            d = nn.ReLU()(out[:, :self.n_d])
            steps_output.append(d)
            # update attention
            attn = out[:, self.n_d:]

        M_loss /= self.n_steps
        return steps_output, M_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)

        prior = torch.ones(x.shape)
        M_explain = torch.zeros(x.shape)
        attn = self.initial_splitter(x)[:, self.n_d:]

        masks = {}
        for step in range(self.n_steps):
            M = self.attn_transformers[step](prior, attn)
            masks[step] = M
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feature_transformers[step](masked_x)
            d = nn.ReLU()(out[:, :self.n_d])
            # explain
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M, step_importance.unsqueeze(dim=1))
            # update attention
            attn = out[:, self.n_d:]
        return M_explain, masks


class TabNetDecoder(nn.Module):
    def __init__(self, input_dim: int, n_d=8, n_steps=3,
                 n_independent=1, n_shared=1,
                 virtual_batch_size=128, momentum=0.02):
        super(TabNetDecoder, self).__init__()

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_steps = n_steps
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size

        if self.n_shared > 0:
            shared_feature_transform = nn.ModuleList(
                [
                    nn.Linear(n_d, 2 * n_d, bias=False)
                    for _ in range(self.n_shared)
                ]
            )
        else:
            shared_feature_transform = None

        self.feature_transformers = nn.ModuleList(
            [
                FeatureTransformer(
                    n_d, n_d, shared_feature_transform,
                    n_glu_independent=self.n_independent,
                    virtual_batch_size=self.virtual_batch_size,
                    momentum=momentum
                ) for _ in range(n_steps)
            ]
        )

        self.reconstruction_layer = nn.Linear(n_d, self.input_dim, bias=False)
        initialize_non_glu(self.reconstruction_layer, n_d, self.input_dim)

    def forward(self, steps_output):
        res = 0
        for step_nb, step_output in enumerate(steps_output):
            x = self.feature_transformers[step_nb](step_output)
            res = torch.add(res, x)
        res = self.reconstruction_layer(res)
        return res


class TabNetPretraining(nn.Module):
    def __init__(self, input_dim: int, pretraining_ratio=0.2,
                 n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02,
                 mask_type='sparsemax', n_shared_decoder=1,
                 n_indep_decoder=1):
        super(TabNetPretraining, self).__init__()

        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.pretraining_ratio = pretraining_ratio
        self.n_shared_decoder = n_shared_decoder
        self.n_indep_decoder = n_indep_decoder

        if self.n_steps <= 0:
            raise ValueError('n_steps should be a positive integer.')
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.masker = RandomObfuscator(self.pretraining_ratio)
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )
        self.decoder = TabNetDecoder(
            self.post_embed_dim,
            n_d=n_d,
            n_steps=n_steps,
            n_independent=self.n_indep_decoder,
            n_shared=self.n_shared_decoder,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

    def forward(self, x):
        embedded_x = self.embedder(x)
        if self.training:
            masked_x, obf_vars = self.masker(embedded_x)
            prior = 1 - obf_vars
            steps_out, _ = self.encoder(masked_x, prior=prior)
            res = self.decoder(steps_out)
            return res, embedded_x, obf_vars
        else:
            steps_out, _ = self.encoder(embedded_x)
            res = self.decoder(steps_out)
            return res, embedded_x, torch.ones(embedded_x.shape)

    def forward_masks(self, x):
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)


class TabNetNoEmbeddings(nn.Module):
    def __init__(self, input_dim: int, output_dim: Union[int, List[int]],
                 n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=2,
                 n_shared=2, epsilon=1e-15, virtual_batch_size=128,
                 momentum=0.02, mask_type='sparsemax'):
        super(TabNetNoEmbeddings, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )

        if self.is_multi_task:
            self.multi_task_mappings = nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = nn.Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = nn.Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        steps_output, M_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        if self.is_multi_task:
            # Result will be in list format
            out = [task_mapping(res) for task_mapping in self.multi_task_mappings]
        else:
            out = self.final_mapping(res)
        return out, M_loss

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)


class TabNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02,
                 mask_type='sparsemax'):
        super(TabNet, self).__init__()

        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.tabnet = TabNetNoEmbeddings(
            self.post_embed_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            mask_type
        )

    def forward(self, x):
        x = self.embedder(x)
        return self.tabnet(x)

    def forward_masks(self, x):
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)


class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 virtual_batch_size=128, momentum=0.02,
                 mask_type='sparsemax'):
        super(AttentiveTransformer, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

        if mask_type == 'sparsemax':
            self.selector = sparsemax.Sparsemax(dim=-1)
        elif mask_type == 'entmax':
            self.selector = sparsemax.Entmax15(dim=-1)
        else:
            raise NotImplementedError(
                'Please choose either sparsemax' + 'or entmax as mask_type'
            )

    def forward(self, priors, processed_feature):
        x = self.fc(processed_feature)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 shared_layers: nn.ModuleList, n_glu_independent: int,
                 virtual_batch_size=128, momentum=0.02,
                 ):
        super(FeatureTransformer, self).__init__()

        params = {
            'n_glu': n_glu_independent,
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum,
        }

        if shared_layers is None:
            self.shared = nn.Identity()
            is_first = True
        else:
            self.shared = GLUBlock(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum
            )
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLUBlock(spec_input_dim, output_dim, first=is_first, **params)

    def forward(self, x: Tensor):
        x = self.shared(x)
        x = self.specifics(x)
        return x


class GLUBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 n_glu=2, first=False, shared_layers=None,
                 virtual_batch_size=128, momentum=0.02):
        super(GLUBlock, self).__init__()

        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = nn.ModuleList()

        params = {
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum,
        }

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLULayer(input_dim, output_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLULayer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x: Tensor):
        scale = torch.sqrt(torch.FloatTensor([0.5]))
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


class GLULayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 fc=None, virtual_batch_size=128, momentum=0.02):
        super(GLULayer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        self.bn = GBN(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, :self.output_dim], torch.sigmoid(x[:, self.output_dim:]))
        return out


class EmbeddingGenerator(nn.Module):
    def __init__(self, input_dim: int, cat_dims: List[int]
                 , cat_idxs: List[int], cat_emb_dim: Union[List[int], int]):
        super(EmbeddingGenerator, self).__init__()

        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return
        elif (cat_dims == []) ^ (cat_idxs == []):
            if cat_dims == []:
                msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
            else:
                msg = "If cat_dims is non-empty, cat_dims must be defined as a list of same length."
            raise ValueError(msg)
        elif len(cat_dims) != len(cat_idxs):
            msg = "The lists cat_dims and cat_idxs must have the same length."
            raise ValueError(msg)
        self.skip_embedding = False

        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                                  and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(
            input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims)
        )

        # Sort dims by cat_idxs
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_dim, emb_dim)
                for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims)
            ]
        )

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embeddings to inputs.
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            return x

        cols = []
        cat_feature_counter = 0
        for feature_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feature_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feature_counter](x[:, feature_init_idx].long())
                )
                cat_feature_counter += 1
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings


class RandomObfuscator(nn.Module):
    """
    Create and applies obfuscation masks
    """
    def __init__(self, pretraining_ratio: float):
        super(RandomObfuscator, self).__init__()
        self.pretraining_ratio = pretraining_ratio

    def forward(self, x):
        obfuscated_vars = torch.bernoulli(
            self.pretraining_ratio * torch.ones(x.shape)
        )
        masked_input = torch.mul(1 - obfuscated_vars, x)
        return masked_input, obfuscated_vars


if __name__ == '__main__':
    model = TabNetModel(input_dim=9)
    print(model)
